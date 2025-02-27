import os
import argparse
from utils import *
from tqdm import tqdm
from torch import optim
from model import my_model
import torch.nn.functional as F
import LDA_SLIC_PU
from functions import get_data,normalize,data_process,spixel_to_pixel_labels,cluster_accuracy,get_args,get_args_key,pprint_args,get_dataset
from sklearn.cluster import KMeans


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')  # 400
    parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--sigma', type=float, default=0.01, help='Sigma of gaussian distribution')
    parser.add_argument('--dataset', type=str, default='PaviaU', help='type of dataset.')  # 'Indian', 'Salinas', 'PaviaU'  'Houston','Trento'
    parser.add_argument('--superpixel_scale', type=int, default=160, help="superpixel_scale") # IP 100 sa  250  pu160  Tr900  HU100

    args = parser.parse_args()


    print("Using {} dataset".format(args.dataset))



    input, num_classes, y_true, gt_reshape, gt_hsi = get_data(args.dataset)
    # normalize data by band norm
    input_normalize = normalize(input)
    height, width, band = input_normalize.shape  # 145*145*200
    print("height={0},width={1},band={2}".format(height, width, band))
    input_numpy = np.array(input_normalize)


    ls = LDA_SLIC_PU.LDA_SLIC(input_numpy,gt_hsi, num_classes - 1)
    Q, S, A, Edge_index, Edge_atter, Seg,A_ones = ls.simple_superpixel(args.superpixel_scale)
    A = torch.from_numpy(A).to(device)


    features = S
    true_labels = gt_reshape
    adj = sp.csr_matrix(A_ones)
    args.cluster_num=num_classes


    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()


    best_acc= best_kappa= best_nmi = best_ari= best_pur = 0
    best_ca = []

    for seed in range(10):
        setup_seed(seed)
        # best_acc, best_nmi, best_ari, best_f1, prediect_labels = clustering(sm_fea_s, true_labels, args.cluster_num)
        model = my_model([features.shape[1]] + args.dims)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        inx = sm_fea_s.to(device)
        target = torch.FloatTensor(adj_1st).to(device)

        print('Start Training...')
        for epoch in tqdm(range(args.epochs)):
            model.train()
            z1, z2 = model(inx, is_train=True, sigma=args.sigma)
            S = z1 @ z2.T
            loss = F.mse_loss(S, target)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                model.eval()
                z1, z2 = model(inx, is_train=False, sigma=args.sigma)
                hidden_emb = (z1 + z2) / 2


                kmeans = KMeans(n_clusters=args.cluster_num).fit(hidden_emb.cpu().detach().numpy())
                predict_labels = kmeans.predict(hidden_emb.cpu().detach().numpy())

                indx = np.where(gt_reshape != 0)
                labels = gt_reshape[indx]

                pixel_y = spixel_to_pixel_labels(predict_labels, Q)
                prediction = pixel_y[indx]

                acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels, prediction, return_aligned=False)


                if acc >= best_acc:
                    best_acc = acc
                    best_kappa = kappa
                    best_nmi = nmi
                    best_ari = ari
                    best_pur = pur
                    best_ca = ca


        print('k-means --- ACC: %5.4f, NMI: %5.4f' % (best_acc, best_nmi))
        for i in range(args.cluster_num):
            print('class_%d:' % (i + 1), end='')
            print('(%.2f)' % (((np.where(gt_reshape == i + 1)[0]).shape[0]) / ((indx[0]).shape[0]) * 100), end=' ')
            print(best_ca[i])


        tqdm.write('acc: {}, nmi: {}, ari: {}'.format(best_acc, best_nmi, best_ari))
        file = open("result_baseline.csv", "a+")
        print(best_acc, best_nmi, best_ari, file=file)
        file.close()

        f = open('./results/' + args.dataset + '_results.txt', 'a+')
        str_results = '\n\n************************************************' \
                      + '\nseed: %s' % (str(seed)) \
                      + '\nacc={:.4f}'.format(best_acc) \
                      + '\nkappa={:.4f}'.format(best_kappa) \
                      + '\nnmi={:.4f}'.format(best_nmi) \
                      + '\nari={:.4f}'.format(best_ari) \
                      + '\npur={:.4f}'.format(best_pur) \
                      + '\nca=' + str(np.around(best_ca, 4)) \


        f.write(str_results)
        f.close()



