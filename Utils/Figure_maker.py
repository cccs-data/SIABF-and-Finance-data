import matplotlib.pyplot as plt

def Figure_Appendx(n, sub_n, pre_len, x_1, x_2, train_len, t_len, lab):
    font_size = 25
    legend_font_size = 20
    line_width = 3
    plt.subplot(n, 1, sub_n)
    # plt.figure(figsize=[20,4])
    plt.plot(pre_len, x_1[train_len:t_len], label='Ground Truth', linewidth=line_width)
    plt.plot(pre_len, x_2[train_len:t_len], label=lab, linewidth=line_width)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.title('('+ str(chr(96+sub_n)) +')', fontsize=font_size+10, x=-0.05, y=1)
    plt.legend(loc='upper right', fontsize=legend_font_size)
    if sub_n == n:
        plt.xlabel('Predict Length', fontsize=font_size)
