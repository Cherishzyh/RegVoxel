import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

color_list = sns.color_palette(['#e50000', '#fffd01', '#87fd05', '#00ffff', '#152eff', '#ff08e8', '#ff5b00', '#9900fa']) \
             + sns.color_palette('deep')


def ShowLossCurve(loss_dict):
    plt.xlabel("epoch") #x轴标签
    plt.ylabel("loss value") #y轴标签
    for index, key in enumerate(loss_dict.keys()):
        y = loss_dict[key]
        x = np.arange(1, len(y)+1)
        plt.plot(y, x, c=color_list[index], label=key)

    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    b = {'a': [1, 4, 6], 'b': [5, 8, 7]}
    ShowLossCurve(b)