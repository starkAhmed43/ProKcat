import torch
import torch.nn as nn
import torch.nn.functional as F


# NOTE: Class blocks below are adapted from authors' notebook implementation:
# code/run_train_test.ipynb, cell id #VSC-33723049.
# We intentionally keep architecture behavior close to notebook code.


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        wh = torch.matmul(h, self.W)
        a_input = self._prepare_attentional_mechanism_input(wh)
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(3), self.alpha)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        h_prime = torch.bmm(attention, wh)

        return F.elu(h_prime) if self.concat else h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        b = wh.size()[0]
        n = wh.size()[1]

        wh_repeated_in_chunks = wh.repeat_interleave(n, dim=1)
        wh_repeated_alternating = wh.repeat_interleave(n, dim=0).view(b, n * n, self.out_features)
        all_combinations_matrix = torch.cat([wh_repeated_in_chunks, wh_repeated_alternating], dim=2)

        return all_combinations_matrix.view(b, n, n, 2 * self.out_features)


class ProKcatBackboneForMLP(nn.Module):
    """
    Notebook-derived backbone that returns intermediate features
    (cf, fps, pf, inverse_Temp, Temperature), exactly like the notebook's Plan-3 path.
    """

    def __init__(self, n_atom, n_amino, comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out):
        super().__init__()

        self.embedding_layer_atom = nn.Embedding(n_atom + 1, comp_dim)
        self.embedding_layer_amino = nn.Embedding(n_amino + 1, prot_dim)

        self.dropout = dropout
        self.alpha = alpha
        self.layer_cnn = layer_cnn
        self.latent_dim = latent_dim
        self.layer_out = layer_out

        self.gat_layers = [GATLayer(comp_dim, gat_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_head)]
        for i, layer in enumerate(self.gat_layers):
            self.add_module(f"gat_layer_{i}", layer)
        self.gat_out = GATLayer(gat_dim * num_head, comp_dim, dropout=dropout, alpha=alpha, concat=False)
        self.W_comp = nn.Linear(comp_dim, latent_dim)

        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * window + 1, stride=1, padding=window) for _ in range(layer_cnn)]
        )
        self.W_prot = nn.Linear(prot_dim, latent_dim)

        self.fp0 = nn.Parameter(torch.empty(size=(1024, latent_dim)))
        nn.init.xavier_uniform_(self.fp0, gain=1.414)
        self.fp1 = nn.Parameter(torch.empty(size=(latent_dim, latent_dim)))
        nn.init.xavier_uniform_(self.fp1, gain=1.414)

        self.bidat_num = 4
        self.U = nn.ParameterList([nn.Parameter(torch.empty(size=(latent_dim, latent_dim))) for _ in range(self.bidat_num)])
        for i in range(self.bidat_num):
            nn.init.xavier_uniform_(self.U[i], gain=1.414)

        self.transform_c2p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.transform_p2c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])

        self.bihidden_c = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.bihidden_p = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for _ in range(self.bidat_num)])
        self.biatt_c = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])
        self.biatt_p = nn.ModuleList([nn.Linear(latent_dim * 2, 1) for _ in range(self.bidat_num)])

        self.comb_c = nn.Linear(latent_dim * self.bidat_num, latent_dim)
        self.comb_p = nn.Linear(latent_dim * self.bidat_num, latent_dim)

    def comp_gat(self, atoms, adj):
        atoms_vector = self.embedding_layer_atom(atoms)
        atoms_multi_head = torch.cat([gat(atoms_vector, adj) for gat in self.gat_layers], dim=2)
        atoms_vector = F.elu(self.gat_out(atoms_multi_head, adj))
        atoms_vector = F.leaky_relu(self.W_comp(atoms_vector), self.alpha)
        return atoms_vector

    def prot_cnn(self, amino):
        amino_vector = self.embedding_layer_amino(amino)
        amino_vector = torch.unsqueeze(amino_vector, 1)
        for i in range(self.layer_cnn):
            amino_vector = F.leaky_relu(self.conv_layers[i](amino_vector), self.alpha)
        amino_vector = torch.squeeze(amino_vector, 1)
        amino_vector = F.leaky_relu(self.W_prot(amino_vector), self.alpha)
        return amino_vector

    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
        return a_softmax

    def bidirectional_attention_prediction(self, atoms_vector, atoms_mask, fps, amino_vector, amino_mask, inv_temp, temp):
        b = atoms_vector.shape[0]
        for i in range(self.bidat_num):
            A = torch.tanh(torch.matmul(torch.matmul(atoms_vector, self.U[i]), amino_vector.transpose(1, 2)))
            A = A * torch.matmul(atoms_mask.view(b, -1, 1), amino_mask.view(b, 1, -1))

            atoms_trans = torch.matmul(A, torch.tanh(self.transform_p2c[i](amino_vector)))
            amino_trans = torch.matmul(A.transpose(1, 2), torch.tanh(self.transform_c2p[i](atoms_vector)))

            atoms_tmp = torch.cat([torch.tanh(self.bihidden_c[i](atoms_vector)), atoms_trans], dim=2)
            amino_tmp = torch.cat([torch.tanh(self.bihidden_p[i](amino_vector)), amino_trans], dim=2)

            atoms_att = self.mask_softmax(self.biatt_c[i](atoms_tmp).view(b, -1), atoms_mask.view(b, -1))
            amino_att = self.mask_softmax(self.biatt_p[i](amino_tmp).view(b, -1), amino_mask.view(b, -1))

            cf = torch.sum(atoms_vector * atoms_att.view(b, -1, 1), dim=1)
            pf = torch.sum(amino_vector * amino_att.view(b, -1, 1), dim=1)

            if i == 0:
                cat_cf = cf
                cat_pf = pf
            else:
                cat_cf = torch.cat([cat_cf.view(b, -1), cf.view(b, -1)], dim=1)
                cat_pf = torch.cat([cat_pf.view(b, -1), pf.view(b, -1)], dim=1)

        inverse_temp = inv_temp.view(inv_temp.shape[0], -1)
        temperature = temp.view(temp.shape[0], -1)

        # Notebook plan-3 output path.
        cf = self.comb_c(cat_cf).view(b, -1)
        fps = fps.view(b, -1)
        pf = self.comb_p(cat_pf)
        return cf, fps, pf, inverse_temp, temperature

    def forward(self, atoms, atoms_mask, adjacency, amino, amino_mask, fps, inv_temp, temp):
        atoms_vector = self.comp_gat(atoms, adjacency)
        amino_vector = self.prot_cnn(amino)

        super_feature = F.leaky_relu(torch.matmul(fps, self.fp0), 0.1)
        super_feature = F.leaky_relu(torch.matmul(super_feature, self.fp1), 0.1)

        return self.bidirectional_attention_prediction(
            atoms_vector,
            atoms_mask,
            super_feature,
            amino_vector,
            amino_mask,
            inv_temp,
            temp,
        )


class NewMLPIndependent(nn.Module):
    # NOTE: Adapted from run_train_test.ipynb, class New_MLP_Independent.
    def __init__(self, alpha, latent_dim):
        super().__init__()
        self.alpha = alpha
        self.W_final_1 = nn.ModuleList([nn.Linear(latent_dim * 3, latent_dim * 3) for _ in range(3)])
        self.W_final_2 = nn.Linear(latent_dim * 3, 3)
        self.W_final_3 = nn.Linear(5, 1)

    def forward(self, cf, fps, pf, inverse_temp, temperature):
        cat_vector = torch.cat((cf, fps, pf), dim=1)
        for j in range(3):
            cat_vector = F.leaky_relu(self.W_final_1[j](cat_vector), self.alpha)

        cat_vector = F.leaky_relu(self.W_final_2(cat_vector), self.alpha)
        cat_vector = torch.cat((cat_vector, inverse_temp, temperature), dim=1)
        return self.W_final_3(cat_vector), cat_vector
