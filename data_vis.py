# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:10:13 2020

@author: lq
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns 
os.chdir('C:\\Users\\Administrator\\Desktop\\外接\\data')

#%% 读取清洗后的数据
dataset = pd.read_csv('new_data_clearned.csv')
#%% 数据清洗  
def data_clearned(data):
    dataset = pd.read_csv('result.csv')
    terminal = dataset #dataset.head(10000)
    # Time Duration 为零的过滤掉
    terminal_one = terminal[terminal['Time Duration'] != 0 ]
    # Duration Level 为5->? 的过滤掉
    terminal_two = terminal_one[~terminal_one['Duration Level'].str.contains('5->')]
    # 合并 同状态转移
    terminal_three = terminal_two.groupby(['Duration Level','Hashcode'])['Time Duration'].sum().reset_index()
    # 拼接 columns = ['Duration Level'， ‘Hashcode’， ‘Time Duration’, 'Gender', 'Age']
    terminal_four = terminal.groupby('Hashcode')[['Age','Gender']].max().reset_index()
    terminal_clearned = pd.merge(terminal_three, terminal_four,how='left', on='Hashcode')
    terminal_clearned.tocsv('data_clearned.csv')
    return terminal_clearned

#%% for new_data.csv
def data_clearned(data):
    # dataset = pd.read_csv('result.csv')
    terminal = data #dataset.head(10000)
    # Time Duration 为零的过滤掉
    terminal = terminal[terminal['Time Duration'] != 0 ]
    # Duration Level 为5->? 的过滤掉
    terminal = terminal[~terminal['Duration Level'].str.contains('5->')]
    terminal_clearned.tocsv('new_data_clearned.csv')
    return terminal_clearned


#%% 拆分Duration_Level (1->2)
def change_Duration_Level(df):
    pre_level = []
    next_level = []
    for i in df['Duration Level'].str.split('->').values:
        pre_level.append(int(i[0]))
        next_level.append(int(i[1]))
    df['pre_level'] = pre_level
    df['next_level'] = next_level
change_Duration_Level(dataset)
dataset_copy = dataset.copy()
duration_level_count = dataset_copy.groupby('Duration Level')['Age'].count() 
#%% 不同状态下数量的分布（按数量和概率）, 四张图合并在一张图内
def question_one():
    scope = [0,4,8,12,16]  # different status transation index in duration_level_count
    x = duration_level_count.index
    x = [i.replace('->','-') for i in x]
    y = duration_level_count.values.tolist()
    plt.figure(figsize=(10, 12))
    for i in range(4):
        ax = plt.subplot(221+i)
        # ax.set_title('%.3f' % alpha)
        x_ = x[scope[i]:scope[i+1]]
        y_ = y[scope[i]:scope[i+1]]
        plt.title("The Distribution of Transitional Duration")
        plt.xlabel("Status trasition",fontsize=10)
        plt.ylabel("Num")
        plt.bar(x=x_, height=y_, width=0.3)
        for x__, y__ in zip(x_, y_):
            plt.text(x__, y__ + 1, str(y__), ha='center', va='bottom', fontsize=10, rotation=0)
    plt.show()
    
def question_one_():
    scope = [0,4,8,12,16]  # different status transation index in duration_level_count
    x = duration_level_count.index
    x = [i.replace('->','-') for i in x]
    y = duration_level_count.values.tolist()
    plt.figure(figsize=(10, 12))
    for i in range(4):
        ax = plt.subplot(221+i)
        # ax.set_title('%.3f' % alpha)
        x_ = x[scope[i]:scope[i+1]]
        y_ = y[scope[i]:scope[i+1]]
        plt.title("The Distribution of Transitional Duration")
        y_p = [i/sum(y_) for i in y_]
        # plt.bar(x=x_, height=y_, width=0.3)
        colors=['#9999ff','#ff9999','#7777aa','#2442aa','#dd5555']
        labels = x_
        plt.pie(x = y_p, # 绘图数据
                # explode=explode, # 突出显示大专人群
                labels=labels, # 添加教育水平标签
                colors=colors, # 设置饼图的自定义填充色
                autopct='%.1f%%', # 设置百分比的格式，这里保留一位小数
                pctdistance=0.8,  # 设置百分比标签与圆心的距离
                labeldistance = 1.1, # 设置教育水平标签与圆心的距离
                startangle = 180, # 设置饼图的初始角度
                radius = 1.2, # 设置饼图的半径
                counterclock = False, # 是否逆时针，这里设置为顺时针方向
                wedgeprops = {'linewidth': 1.5, 'edgecolor':'green'},# 设置饼图内外边界的属性值
                textprops = {'fontsize':10, 'color':'black'}, # 设置文本标签的属性值
                )

    plt.show()    
    
question_one_()
#%%
def duration_time_distrition(data, category=0, num=5):
    if category == 0: # 策略0 均值划分  左闭右开
        max_val = max(data)
        min_val = min(data)
        d = math.ceil((max_val - min_val) / num)
        steps = [[min_val+i*d,min_val+(i+1)*(d-1)] for i in range(num) ]
        steps[num-1][1] = max_val
        count = [0] * num
        for i in data:
            try:
                count[math.floor((i-min_val)/d)] += 1                        
            except:
                print(num, math.floor((i-min_val)/d), min_val, max_val, d, i)
        return steps, count

def duration_time_distrition_(data, category=0, num=5,d=100):
    if category == 0: # 策略0 均值划分  左闭右开
        steps = [[0,50],[50,100],[100,150],[150,200],[200,250],[250,300],[300,350],[350,400],[400,2000]]
        count = [0] * len(steps)
        for i in data:
            for idx,j in enumerate(steps):
                if i < j[1]:
                    count[idx] += 1
                    break
        return steps, count
status_trasition_data = {i:dataset[dataset['Duration Level'] == i] for i in duration_level_count.index}
def compute_res():
    res = {}
    res_gender = {}
    res_age = {}
    # 计算时间状态下的时间分布数量
    for key in status_trasition_data:
        dt = status_trasition_data[key]['Time Duration'].values.tolist()
        res[key] = duration_time_distrition_(dt,d=100)
    
    # 计算时间状态下的时间分布数量，考虑性别
    for key in status_trasition_data:
        df =  status_trasition_data[key]
        df_f = df[df['Gender'] == 'F']
        df_m = df[df['Gender'] == 'M']
        dt_f = df_f['Time Duration'].values.tolist()
        dt_m = df_m['Time Duration'].values.tolist()
        res_gender[key] = {'M':duration_time_distrition_(dt_m), 'F':duration_time_distrition_(dt_f)}

    # 计算时间状态下的时间分布数量，考虑年龄 0-5,5-18,18-45,45-75,75
    for key in status_trasition_data:
        df = status_trasition_data[key]
        df_baby = df[(0<=df['Age']) & (df['Age']<5)]['Time Duration'].values.tolist()
        df_youth = df[(5<=df['Age']) & (df['Age']<18)]['Time Duration'].values.tolist()
        df_adult = df[(18<=df['Age']) & (df['Age']<45)]['Time Duration'].values.tolist()
        df_middle = df[(45<=df['Age']) & (df['Age']<65)]['Time Duration'].values.tolist()
        df_middle_plus = df[(65<=df['Age']) & (df['Age']<75)]['Time Duration'].values.tolist()
        df_old = df[(75<=df['Age'])]['Time Duration'].values.tolist()
        res_age[key] = {'baby':duration_time_distrition_(df_baby), 
                        'youth':duration_time_distrition_(df_youth),
                        'adult':duration_time_distrition_(df_adult),
                        'middle':duration_time_distrition_(df_middle),
                        'middle_plus':duration_time_distrition_(df_middle_plus),
                        'old':duration_time_distrition_(df_old)
                        }
    return res, res_gender, res_age
res, res_gender, res_age = compute_res()  
#%% 不同状态的时间分布
def execute_before():
    all_pos = []
    num = 1
    p = 0
    # 获取每张图的pos
    for idx,key in enumerate(list(status_trasition_data.keys())):
        k = {}
        k['key'] = key
        row, col = key.split('->')
        if int(row) != num:
            num += 1
            p = 0
        
        k['pos'] = (num-1) * 5 + p
        p += 1
        all_pos.append(k)
    return all_pos
# 画图
def question_two_three_four_five(strategy_code=0):
    all_pos = execute_before()
    plt.figure(figsize=(37, 35)) 
    for i in all_pos:   
        ax = plt.subplot(4,5,i['pos']+1)
        plt.ylabel("frequency")
        plt.xlabel("Duration time", fontsize=8)  
        title = i['key'].replace('>','')
        plt.title(f'Status trasition:{title}',fontsize=10)
        # 不同状态转移的时间分布
        if strategy_code == 2: 
            ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
            ax.set_ylim([0,1])            
            steps, count = res[i['key']]
            ax.set_xticks(range(len(steps)))
            xticklabels = []
            for idx, k in enumerate(steps):
                x = str(k)
                if idx != len(steps) -1:
                    x = x[:-1] +  ')'
                xticklabels.append(x)            
            y_ = count
            y_ = [j/sum(y_) for j in y_]
            m = {i:count[idx] for idx,i in enumerate(y_)}
            x_ = np.arange(len(y_))
            ax.bar(x=x_, height=y_, width=0.3)
            ax.set_xticklabels(xticklabels, fontsize=7)
            for x__, y__ in zip(x_, y_):
                plt.text(x__, y__, str(m[y__]), ha='center', va='bottom', fontsize=10, rotation=0)
        # 不同时间状态的时间分布(按性别分)
        elif strategy_code == 3:
            ax = plt.subplot(4,5,i['pos']+1)
            ax.set_yticks([0,0.2,0.4,0.6])
            ax.set_ylim([0,0.6])
            count_gender = res_gender[i['key']]
            steps, count_f = count_gender['F']
            steps_, count_m = count_gender['M']
            ax.set_xticks(range(len(steps)))
            xticklabels = []
            for idx, k in enumerate(steps):
                x = str(k)
                if idx != len(steps) -1:
                    x = x[:-1] +  ')'
                xticklabels.append(x)
            y_f = np.array([j/sum(count_m+count_f) for j in count_f])
            y_m = np.array([j/sum(count_m+count_f) for j in count_m])
            m_f = {i:count_f[idx] for idx,i in enumerate(y_f)}
            m_m = {i:count_m[idx] for idx,i in enumerate(y_m)}
            x_ = np.arange(len(y_f))
           
            
            # ax.bar(x=x_, height=y_m, width=0.3, color='#1F77B4',label='M')
            # ax.bar(x=x_+0.31, height=y_f, width=0.3, color='#AEC6E7',label='F')
            ax.bar(x=x_, height=y_f, width=0.3, color='#AEC6E7',label='F')
            # plt.bar(x=x_, height=y_m, width=0.3, color='#1F77B4',label='M')
            # plt.legend()
            # plt.bar(x=x_+0.31, height=y_f, width=0.3, color='#AEC6E7',label='F')     
            # plt.legend()
            ax.legend()
            ax.set_xticklabels(xticklabels, fontsize=7)  
                               
        # 不同时间状态的时间分布(按年龄分)
        elif strategy_code == 4:
            ax.set_yticks([0,0.2,0.4,0.6,0.8])
            ax.set_ylim([0,0.65])            
            count_age = res_age[i['key']]
            steps, count_baby = count_age['baby']
            _, count_youth = count_age['youth']
            _, count_adult = count_age['adult']
            _, count_middle = count_age['middle']
            _, count_middle_plus = count_age['middle_plus']
            _, count_old = count_age['old']
            ax.set_xticks(range(len(steps)))
            xticklabels = []
            for idx, k in enumerate(steps):
                x = str(k)
                if idx != len(steps) -1:
                    x = x[:-1] +  ')'
                xticklabels.append(x)  
            s = sum(count_baby+count_youth+count_adult+count_middle+count_old)
            y_baby = np.array([j/s for j in count_baby])
            y_youth = np.array([j/s for j in count_youth])
            y_adult = np.array([j/s for j in count_adult]) 
            y_middle = np.array([j/s for j in count_middle]) 
            y_middle_plus = np.array([j/s for j in count_middle_plus]) 
            y_old = np.array([j/s for j in count_old]) 
            m_baby = {i:count_baby[idx] for idx,i in enumerate(y_baby)}
            m_youth = {i:count_youth[idx] for idx,i in enumerate(y_youth)}
            m_adult = {i:count_adult[idx] for idx,i in enumerate(y_adult)}
            m_middle = {i:count_middle[idx] for idx,i in enumerate(y_middle)}
            m_middle_plus = {i:count_middle[idx] for idx,i in enumerate(y_middle_plus)}
            m_old = {i:count_old[idx] for idx,i in enumerate(y_old)}
            x_ = np.arange(len(y_baby))
            # ax.bar(x=x_, height=y_baby, width=0.5,color='#302624',label='[0,5)')
            # ax.bar(x=x_, height=y_youth, bottom=y_baby, width=0.5, color='#C5B0D5', label='[5,18)')
            # ax.bar(x=x_, height=y_adult, bottom=y_youth, width=0.5, color='#98DF8A', label='[18,45)')
            # ax.bar(x=x_, height=y_middle, bottom=y_adult, width=0.5, color='#C49C94', label='[45,65)')
            # ax.bar(x=x_, height=y_middle_plus, bottom=y_middle, width=0.5, color='#AEC6E7', label='[65,75)')
            # ax.bar(x=x_, height=y_old, bottom=y_middle_plus, width=0.5, color='#FFBB78', label='[75, )')
            
            # ax.bar(x=x_-0.24, height=y_baby, width=0.12, color='#302624',label='[0,5)')
            # ax.bar(x=x_-0.12, height=y_youth, width=0.12, color='#C5B0D5',label='[5,18)') 
            # ax.bar(x=x_, height=y_adult, width=0.12, color='#98DF8A',label='[18,45)')
            # ax.bar(x=x_+0.12, height=y_middle, width=0.12, color='#C49C94',label='[45,65)')
            # ax.bar(x=x_+0.24, height=y_middle_plus, width=0.12, color='#AEC6E7',label='[65,75)')
            ax.bar(x=x_, height=y_old, width=0.12, color='#FFBB78',label='[75, )')
            
            ax.set_xticklabels(xticklabels, fontsize=7)               
            ax.legend()            
        # 拟合曲线
        elif strategy_code == 5:
            ax.set_yticks([0,0.01,0.02,0.03,0.04,0.05])
            ax.set_ylim([0,0.07])  
            x = status_trasition_data[i['key']]['Time Duration']
            nbins = 30
            freq, bins = np.histogram(x, bins=nbins)
            sns.distplot(x,bins=nbins,
                         hist=True, # Whether to plot a (normed) histogram.
                         kde=True, 
                         ax=ax,
                         norm_hist=False, # norm_hist = norm_hist or kde or (fit is not None); 如果为False且kde=False, 则高度为频数
                         rug = False
                         )
            
    plt.show()

# question_two_three_four_five(strategy_code=5)
question_two_three_four_five(4)
#%% 
question_one()
question_two_three_four_five(2)
question_two_three_four_five(3)
question_two_three_four_five(4)
question_two_three_four_five(5)
    
    