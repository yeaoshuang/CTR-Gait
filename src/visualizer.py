import logging, numpy as np, seaborn as sns, matplotlib.pyplot as plt

from . import utils as U


class Visualizer():
    def __init__(self, args):
        self.args = args
        U.set_logging(args)
        logging.info('')
        logging.info('Starting visualizing ...')

        self.action_names = {}
        self.action_names['bone'] = [
            'action 1', 'action 2', 'action 3', 'action 4'
        ]
        self.action_names['joint'] = [
            'action 1', 'action 2', 'action 3', 'action 4'
        ]
        self.action_names['motion'] = [
            'action 1', 'action 2', 'action 3', 'action 4'
        ]
        self.action_names['cmu'] = [
            'walking 1', 'running 2', 'directing_traffic 3', 'soccer 4',
            'basketball 5', 'washwindow 6', 'jumping 7', 'basketball_signal 8'
        ]

        self.font_sizes = {
            'bone': 12,
            'joint': 12,
            'motion': 12,
            'cmu': 14,
        }


    def start(self):
        self.read_data()
        logging.info('Please select visualization function from follows: ')
        logging.info('1) wrong sample (ws), 2) important joints (ij), 3) skeleton (s),')
        logging.info('4) confusion matrix (cm), 5) action accuracy (ac)')
        while True:
            logging.info('Please input the number (or name) of function, q for quit: ')
            cmd = input(U.get_current_timestamp())
            if cmd in ['q', 'quit', 'exit', 'stop']:
                break
            elif cmd == '1' or cmd == 'ws' or cmd == 'wrong sample':
                self.show_wrong_sample()
            elif cmd == '2' or cmd == 'ij' or cmd == 'important joints':
                self.show_important_joints()
            elif cmd == '3' or cmd == 's' or cmd == 'skeleton':
                self.show_skeleton()
            elif cmd == '4' or cmd == 'cm' or cmd == 'confusion matrix':
                self.show_confusion_matrix()
            elif cmd == '5' or cmd == 'ac' or cmd == 'action accuracy':
                self.show_action_accuracy()
            else:
                logging.info('Can not find this function!')
                logging.info('')


    def read_data(self):
        logging.info('Reading data ...')
        logging.info('')
        data_file = './visualization/extraction_{}.npz'.format(self.args.config)
        try:
            data = np.load(data_file)
        except:
            data = None
            logging.info('')
            logging.error('Error: Wrong in loading this extraction file: {}!'.format(data_file))
            logging.info('Please extract the data first!')
            raise ValueError()
        logging.info('*********************Video Name************************')
        logging.info(data['name'][self.args.visualization_sample])
        logging.info('')

        feature = data['feature'][self.args.visualization_sample]
        self.location = data['location']
        if len(self.location) > 0:
            self.location = self.location[self.args.visualization_sample]
        self.data = data['data'][self.args.visualization_sample]
        self.label = data['label']
        weight = data['weight']
        out = data['out']
        cm = data['cm']
        self.cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

        dataset = self.args.dataset
        self.names = self.action_names[dataset]
        self.font_size = self.font_sizes[dataset]

        self.pred = np.argmax(out, 1)
        self.pred_class = self.pred[self.args.visualization_sample] + 1
        self.actural_class = self.label[self.args.visualization_sample] + 1
        if self.args.visualization_class == 0:
            self.args.visualization_class = self.actural_class
        self.probablity = out[self.args.visualization_sample, self.args.visualization_class-1]
        self.result = np.einsum('kc,ctvm->ktvm', weight, feature)   # CAM method
        self.result = self.result[self.args.visualization_class-1]


    def show_action_accuracy(self):
        cm = self.cm.round(4)

        logging.info('Accuracy of each class:')
        accuracy = cm.diagonal()
        for i in range(len(accuracy)):
            logging.info('{}: {}'.format(self.names[i], accuracy[i]))
        logging.info('')

        plt.figure()
        plt.bar(self.names, accuracy, align='center')
        plt.xticks(fontsize=10, rotation=90)
        plt.yticks(fontsize=10)
        plt.show()


    def show_confusion_matrix(self):
        cm = self.cm.round(2)
        show_name_x = [str(i) for i in range(1,len(self.names)+1)]  # Convert to string list
        show_name_y = self.names

        plt.figure()
        font_size = self.font_size
        sns.heatmap(cm, cmap="Blues", annot=True, annot_kws={'fontsize':font_size-2}, cbar=False,
                    square=True, linewidths=0.1, linecolor='black', xticklabels=show_name_x, yticklabels=show_name_y)
        plt.xticks(fontsize=font_size, rotation=0)
        plt.yticks(fontsize=font_size)
        plt.xlabel('Index of Predict Classes', fontsize=font_size)
        plt.ylabel('Index of True Classes', fontsize=font_size)
        plt.show()


    def show_skeleton(self):
        if len(self.location) == 0:
            logging.info('This function is only for skeleton dataset!')
            logging.info('')
            return

        C, T, V, M = self.location.shape
        # Dataset connecting joint definition
        connecting_joint = np.array([2,1,12,3,15,5,6,7,1,9,10,11,1,14,15,16,1,17,18,19,2,20,8,21,12])
        result = np.maximum(self.result, 0)
        result = result/np.max(result)

        if len(self.args.visualization_frames) > 0:
            pause, frames = 10, self.args.visualization_frames
        else:
            pause, frames = 0.1, range(self.location.shape[1])

        plt.figure()
        plt.ion()
        for t in frames:
            if np.sum(self.location[:,t,:,:]) == 0:
                break

            plt.cla()
            plt.xlim(-50, 2000)
            plt.ylim(-50, 1100)
            plt.axis('off')
            plt.title('sample:{}, class:{}, frame:{}\n probablity:{:2.2f}%, pred_class:{}, actural_class:{}'.format(
                self.args.visualization_sample, self.names[self.args.visualization_class-1],
                t, self.probablity*100, self.pred_class, self.actural_class
            ))

            for m in range(M):
                x = self.location[0,t,:,m]
                y = 1080 - self.location[1,t,:,m]

                c = []
                for v in range(V):
                    r = result[t//4,v,m]
                    g = 0
                    b = 1 - r
                    c.append([r,g,b])
                    k = connecting_joint[v] - 1
                    plt.plot([x[v],x[k]], [y[v],y[k]], '-o', c=np.array([0.1,0.1,0.1]), linewidth=0.5, markersize=0)
                plt.scatter(x, y, marker='o', c=c, s=16)
            plt.pause(pause)
        plt.ioff()
        plt.show()


    def show_wrong_sample(self):
        wrong_sample = []
        for i in range(len(self.pred)):
            if not self.pred[i] == self.label[i]:
                wrong_sample.append(i)
        logging.info('*********************Wrong Sample**********************')
        logging.info(wrong_sample)
        logging.info('')


    def show_important_joints(self):
        first_sum = np.sum(self.result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        logging.info('*********************First Person**********************')
        logging.info('Weights of all joints:')
        logging.info(first_sum)
        logging.info('')
        logging.info('Most important joints:')
        logging.info(first_index)
        logging.info('')

        if self.result.shape[-1] > 1:
            second_sum = np.sum(self.result[:,:,1], axis=0)
            second_index = np.argsort(-second_sum) + 1
            logging.info('*********************Second Person*********************')
            logging.info('Weights of all joints:')
            logging.info(second_sum)
            logging.info('')
            logging.info('Most important joints:')
            logging.info(second_index)
            logging.info('')