from .layers import *
import torch
import torch.nn as nn


class RIPGeo(nn.Module):
    def __init__(self, dim_in, dim_z, dim_med, dim_out, collaborative_mlp=True):
        super(RIPGeo, self).__init__()

        # RIPGeo
        self.att_attribute = SimpleAttention1(temperature=dim_z ** 0.5,
                                             d_q_in=dim_in,
                                             d_k_in=dim_in,
                                             d_v_in=dim_in + 2,
                                             d_q_out=dim_z,
                                             d_k_out=dim_z,
                                             d_v_out=dim_z)

        if collaborative_mlp:
            self.pred = SimpleAttention2(temperature=dim_z ** 0.5,
                                        d_q_in=dim_in * 3 + 4,
                                        d_k_in=dim_in,
                                        d_v_in=2,
                                        d_q_out=dim_z,
                                        d_k_out=dim_z,
                                        d_v_out=2,
                                        drop_last_layer=False)

        else:
            self.pred = nn.Sequential(
                nn.Linear(dim_z, dim_med),
                nn.ReLU(),
                nn.Linear(dim_med, dim_out)
            )

        self.collaborative_mlp = collaborative_mlp

        # calculate A
        self.gamma_1 = nn.Parameter(torch.ones(1, 1))
        self.gamma_2 = nn.Parameter(torch.ones(1, 1))
        self.gamma_3 = nn.Parameter(torch.ones(1, 1))
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1))

        # transform in Graph
        self.w_1 = nn.Linear(dim_in + 2, dim_in + 2)
        self.w_2 = nn.Linear(dim_in + 2, dim_in + 2)

    def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay):
        """
        :param lm_X: feature of landmarks [..., 30]: 14 attribute + 16 measurement
        :param lm_Y: location of landmarks [..., 2]: longitude + latitude
        :param tg_X: feature of targets [..., 30]
        :param tg_Y: location of targets [..., 2]
        :param lm_delay: delay from landmark to the common router [..., 1]
        :param tg_delay: delay from target to the common router [..., 1]
        :return:
        """

        N1 = lm_Y.size(0)
        N2 = tg_Y.size(0)
        ones = torch.ones(N1 + N2 + 1).cuda()
        lm_feature = torch.cat((lm_X, lm_Y), dim=1)
        tg_feature_0 = torch.cat((tg_X, torch.zeros(N2, 2).cuda()), dim=1)
        router_0 = torch.mean(lm_feature, dim=0, keepdim=True)
        all_feature_0 = torch.cat((lm_feature, tg_feature_0, router_0), dim=0)

        '''
        star-GNN
        properties:
        1. single directed graph: feature of <landmarks> will never be updated.
        2. the target IP will receive from surrounding landmarks from two ways: 
            (1) attribute similarity-based one-hop propagation;
            (2) delay measurement-based two-hop propagation via the common router;
        '''
        # GNN-step 1
        adj_matrix_0 = torch.diag(ones)
        delay_score = torch.exp(-self.gamma_1 * (self.alpha * lm_delay + self.beta))

        rou2tar_score_0 = torch.exp(-self.gamma_2 * (self.alpha * tg_delay + self.beta)).reshape(N2)

        # feature
        _, attribute_score = self.att_attribute(tg_X, lm_X, lm_feature)
        attribute_score = torch.exp(attribute_score)

        adj_matrix_0[N1:N1 + N2, :N1] = attribute_score
        adj_matrix_0[-1, :N1] = delay_score
        adj_matrix_0[N1:N1 + N2:, -1] = rou2tar_score_0

        degree_0 = torch.sum(adj_matrix_0, dim=1)
        degree_reverse_0 = 1.0 / degree_0
        degree_matrix_reverse_0 = torch.diag(degree_reverse_0)

        degree_mul_adj_0 = degree_matrix_reverse_0 @ adj_matrix_0
        step_1_all_feature = self.w_1(degree_mul_adj_0 @ all_feature_0)

        tg_feature_1 = step_1_all_feature[N1:N1 + N2, :]
        router_1 = step_1_all_feature[-1, :].reshape(1, -1)

        # GNN-step 2
        adj_matrix_1 = torch.diag(ones)
        rou2tar_score_1 = torch.exp(-self.gamma_3 * (self.alpha * tg_delay + self.beta)).reshape(N2)
        adj_matrix_1[N1:N1 + N2:, -1] = rou2tar_score_1

        all_feature_1 = torch.cat((lm_feature, tg_feature_1, router_1), dim=0)

        degree_1 = torch.sum(adj_matrix_1, dim=1)
        degree_reverse_1 = 1.0 / degree_1
        degree_matrix_reverse_1 = torch.diag(degree_reverse_1)

        degree_mul_adj_1 = degree_matrix_reverse_1 @ adj_matrix_1
        step_2_all_feature = self.w_2(degree_mul_adj_1 @ all_feature_1)
        tg_feature_2 = step_2_all_feature[N1:N1 + N2, :]

        final_tg_feature = torch.cat((tg_X,
                                      tg_feature_1,
                                      tg_feature_2), dim=-1)

        '''
        predict
        both normal mlp and collaborative mlp are ok, we suggest:
            (1) the number of neighbors > 10: using collaborative mlp
            (2) the number of neighbors < 10: using normal mlp
        '''

        if self.collaborative_mlp:
            y_pred, _ = self.pred(final_tg_feature, lm_X, lm_Y)
            
        else:
            y_pred = self.pred(final_tg_feature)


        return y_pred, final_tg_feature
