import os
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import time
import smtplib
from email.mime.text import MIMEText
from multiprocessing import Process, Queue
import os, re, sys, linecache
import json
import pdb

def cprint(obj, bg_color='r', end='\n'):
    prefix_dict = {
        'r': '\x1b[1;36;41m',
        'g': '\x1b[1;31;42m',
        'y': '\x1b[1;30;43m',
        'b': '\x1b[1;33;44m',
        'm': '\x1b[1;33;45m',
        'o': '\x1b[1;30;46m',
        'w': '\x1b[1;31;47m'
    }
    print(prefix_dict[bg_color] + obj.__str__() + '\x1b[0m', end=end)

def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'w'):
            pass

    def wcprint(self, obj, bg_color='r', end='\n'):
        cprint(obj, bg_color, end)
        with open(self.log_path, 'a') as fp:
            fp.write('%s\n' % obj.__str__())

def get_exclusive_colors(n):
    return cm.rainbow(np.linspace(0, 1, n))


def plot_pos_neg_hist(values, labels, num_bins, title, fig_name, min_value=None, max_value=None, pos_color=None, neg_color=None, label=None, xlabel=None, ylabel=None, xticks=None, yticks=None, axis_min_x=None, axis_max_x=None, axis_min_y=None, axis_max_y=None, font_size_tick=11, font_size=12, figsize=(4, 4)):
    def centroids_of_bin(bins):
        centroids = []
        for i in range(1, len(bins)):
            centroids.append((bins[i-1] + bins[i]) / 2.0)
        return np.array(centroids)

    if pos_color is None:
        pos_color = list(plt.rcParams['axes.prop_cycle'])[0]['color']
    if neg_color is None:
        neg_color = list(plt.rcParams['axes.prop_cycle'])[2]['color']

    pos_values = values[labels == 1]
    pos_mean = np.mean(pos_values)
    neg_values = values[labels == 0]
    neg_mean = np.mean(neg_values)

    max_value = np.max(values) if max_value == None else max_value
    min_value = np.min(values) if min_value == None else min_value
    bins = np.linspace(min_value, max_value, num_bins)

    # plot hist
    pos_n, pos_bins, neg_patches = plt.hist(pos_values, bins=num_bins)
    neg_n, neg_bins, neg_patches = plt.hist(neg_values, bins=num_bins)

    plt.clf()
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.axvline(pos_mean, color=pos_color)
    ax.axvline(neg_mean, color=neg_color)
    ax.fill_between(pos_bins[:-1], pos_n, alpha=0.4, color=pos_color, label='positive')
    ax.fill_between(neg_bins[:-1], neg_n, alpha=0.4, color=neg_color, label='negative')

    if label is not None:
        ax.legend()

    # adjust axes
    x1, x2, y1, y2 = ax.axis()
    if axis_min_x is not None:
        x1 = axis_min_x
    if axis_max_x is not None:
        x2 = axis_max_x
    if axis_min_y is not None:
        y1 = axis_min_y
    if axis_max_y is not None:
        y2 = axis_max_y
    ax.axis(xmin=x1, ymin=y1, xmax=x2, ymax=y2)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(labelsize=font_size_tick)

    ax.set_title(title, fontsize=font_size)
    fig.tight_layout()
    plt.savefig('%s.png' % fig_name)
    plt.clf()


def plot_cluster_scatter(mat_list, title, fig_name, labels=None, xlabel=None, ylabel=None, xticks=None, yticks=None, axis_min_x=None, axis_max_x=None, axis_min_y=None, axis_max_y=None, font_size_tick=11, font_size=12, figsize=(4, 4)):
    plt.clf()
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # plot
    for idx, mat in enumerate(mat_list):
        color = list(plt.rcParams['axes.prop_cycle'])[idx]['color']
        ax.scatter(mat[:,0], mat[:,1], color=color, label=labels[idx], alpha=0.7)

    if labels is not None:
        ax.legend()

    # adjust axes
    x1, x2, y1, y2 = ax.axis()
    if axis_min_x is not None:
        x1 = axis_min_x
    if axis_max_x is not None:
        x2 = axis_max_x
    if axis_min_y is not None:
        y1 = axis_min_y
    if axis_max_y is not None:
        y2 = axis_max_y

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(labelsize=font_size_tick)

    ax.set_title(title, fontsize=font_size)
    fig.tight_layout()
    plt.savefig('%s.png' % fig_name)
    plt.clf()


def plot(x, y, title, fig_name, color=None, label=None, xlabel=None, ylabel=None, xticks=None, yticks=None, axis_min_x=None, axis_max_x=None, axis_min_y=None, axis_max_y=None, font_size_tick=11, font_size=12, figsize=(4, 4)):
    plt.clf()
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    if color is None:
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']

    # plot
    ax.plot(x, y, color=color, label=label)
    if label is not None:
        ax.legend()

    # adjust axes
    x1, x2, y1, y2 = ax.axis()
    if axis_min_x is not None:
        x1 = axis_min_x
    if axis_max_x is not None:
        x2 = axis_max_x
    if axis_min_y is not None:
        y1 = axis_min_y
    if axis_max_y is not None:
        y2 = axis_max_y

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(labelsize=font_size_tick)

    ax.set_title(title, fontsize=font_size)
    fig.tight_layout()
    plt.savefig('%s.png' % fig_name)
    plt.clf()

def plot_y_mat(x, y_mat, title, fig_name, linewidth=0.3, color=None, mean_color=None, label=None, xlabel=None, ylabel=None, xticks=None, yticks=None, axis_min_x=None, axis_max_x=None, axis_min_y=None, axis_max_y=None, font_size_tick=11, font_size=12, figsize=(4, 4)):
    plt.clf()
    plt.style.use('seaborn')
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # adjust axes
    x1, x2, y1, y2 = ax.axis()
    if axis_min_x is not None:
        x1 = axis_min_x
    if axis_max_x is not None:
        x2 = axis_max_x
    if axis_min_y is not None:
        y1 = axis_min_y
    if axis_max_y is not None:
        y2 = axis_max_y
    ax.set_xlim([x1, x2])
    ax.set_ylim([y1, y2])
    ax.axis((x1, x2, y1, y2))

    if color is None:
        color = list(plt.rcParams['axes.prop_cycle'])[0]['color']
    if mean_color is None:
        mean_color = list(plt.rcParams['axes.prop_cycle'])[2]['color']

    # plot
    n = len(y_mat)
    for idx, y in enumerate(y_mat):
        if idx + 1 == n:
            ax.plot(x, y, color=color, label=label, linewidth=linewidth)
        else:
            ax.plot(x, y, color=color, linewidth=linewidth)


    # mean plot
    ax.plot(x, np.mean(y_mat, axis=0), color=mean_color, label='avg')

    if label is not None:
        ax.legend()


    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.tick_params(labelsize=font_size_tick)

    ax.set_title(title, fontsize=font_size)
    fig.tight_layout()
    plt.savefig('%s.png' % fig_name)
    plt.clf()

def random_sleep(start, end):
    rand_float = np.random.rand()
    random_time = start + (rand_float * (end - start))
    time.sleep(random_time)

def send_mail(title, contents, src, passwd, dst):
    msg = '\r\n'.join([
        'From: %s' % src,
        'To: %s' % dst,
        'Subject: %s' % title,
        '',
        '%s' % contents
    ])
    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    s.ehlo()
    s.login(src, passwd)
    s.sendmail(src, [dst], msg)
    s.quit()

def auto_exe_multi_exep(function_ptr, settings):
    proc_list = []
    res_queue = Queue()
    for setting in settings:
        while True:
            try:
                proc = Process(target=function_ptr, args=(res_queue, setting))
                proc.start()
                break
            except Exception as e:
                cprint("Exception occured at setting %s, %s" % (str(setting), str(e)), bg_color='r')
                exc_type, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                lineno = tb.tb_lineno
                filename = f.f_code.co_filename
                linecache.checkcache(filename)
                line=linecache.getline(filename, lineno, f.f_globals)
                cprint('\t{}, {}, {}: {}'.format(filename, lineno, line.strip(), exc_obj))
        proc_list.append(proc)

    for proc in proc_list:
        cprint("waiting ", bg_color='w', end='')
        cprint(proc, bg_color='w')
        proc.join()

    return res_queue

def load_accounts(file_path):
    try:
        with open(file_path) as fp:
            accounts_dict = json.load(fp)
        return accounts_dict
    except Exception as e:
        cprint('Exception occured during load account json')
        return None


