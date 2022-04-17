# -*- coding:utf-8 -*-
# 代码修改来源：https://github.com/EddyGao/PSO
import numpy as np
import matplotlib.pyplot as plt
import math
import heapq

def getweight():
    # 惯性权重
    weight = 1
    return weight

def getlearningrate():
    # 分别是粒子的个体和社会的学习因子，也称为加速常数
    lr = (0.49445,1.49445)
    return lr
def getinertiaweight():
    # 分别是粒子的个体和社会的学习因子，也称为加速常数
    w = [0.49445,0.49445]
    return w
def getmaxgen():
    # 最大迭代次数
    maxgen = 300
    return maxgen

def getsizepop():
    # 种群规模
    sizepop = 50
    return sizepop

def getrangepop():
    # 粒子的位置的范围限制,x、y方向的限制相同
    #rangepop = (-2*math.pi , 2*math.pi)
    rangepop = (-1, 1)
    return rangepop

def getrangespeed():
    # 粒子的速度范围限制
    rangespeed = (-0.5,0.5)
    return rangespeed

### 设置评分函数为dl_score_with_lightmask
from util import dl_score_with_lightmask as func

def initpopvfit(sizepop, dimensions, rangepop, orgimg, contours, model, half=False, pos_20=False,part='OD'): # 粒子个数,维度,范围
    pop = np.zeros((sizepop,dimensions)) # 粒子pop
    v = np.zeros((sizepop,dimensions))   # 速度v
    fitness = np.zeros(sizepop)
    start = 0
    if pos_20 != False:
        #print("使用上次最优20个粒子初始化")
        start = len(pos_20)
        for i in range(start):
            pop[i] = pos_20[i]
            v[i] = -1 + 0.5*np.random.random(dimensions)
        
    for i in range(start, sizepop):
        pop[i] = -1 + 0.5*np.random.random(dimensions)
        v[i] = -1 + 0.5*np.random.random(dimensions)
    fitness = func(pop, orgimg, contours, model, half)
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness  

def grad(pop,GradDict):
    print(pop.size())
    for i in pop.size(0):
        pop[i] = 
def pso(maxgen, sizepop, dimensions, orgimg, contours, model, half=False, pos_20=False,part='OD',init_weight = 0.6,end_weight = 0.1): # 迭代次数,粒子个数,粒子维度
    w = getweight()
    inertia_w = getinertiaweight()
    lr = getlearningrate()
    #maxgen = getmaxgen()
    #sizepop = getsizepop()
    rangepop = getrangepop()
    rangespeed = getrangespeed()
    
    pop,v,fitness = initpopvfit(sizepop, dimensions, rangepop, orgimg, contours, model, half, pos_20,part)
    gbestpop,gbestfitness,pbestpop,pbestfitness = getinitbest(fitness,pop)

    #result = np.zeros(maxgen)
    
    for i in range(maxgen):
        t=0.5
        #速度更新
        inertia_w[0] = ((init_weight - end_weight) * float(maxgen - i)) / (float(maxgen) +0.001)
        #w[1] = (init_weight - end_weight) * float(maxgen - i) / float(maxgen) +0.1
        for j in range(sizepop):
            v_hat= lr[0]*np.random.rand()*(pbestpop[j]-pop[j])+lr[1]*np.random.rand()*(gbestpop-pop[j])
            v[j] = inertia_w[0]*v[j]+ v_hat #+w[1]*v[(j-1)%sizepop]
        
        v[v<rangespeed[0]] = rangespeed[0]
        v[v>rangespeed[1]] = rangespeed[1]

        #粒子位置更新
        for j in range(sizepop):
            #pop[j] += 0.5*v[j]
            pop[j] = t*(0.5*v[j])+(1-t)*pop[j]
           
        pop[pop<rangepop[0]] = rangepop[0]
        pop[pop>rangepop[1]] = rangepop[1]

        #适应度更新
        #for j in range(sizepop):
        #    fitness[j] = func(pop[i], orgimg, contours, model, half)
       

        fitness= func(pop, orgimg, contours, model, half)
       

        #print("fitness:", fitness)
        for j in range(sizepop):
            if fitness[j] > pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()

        if pbestfitness.max() > gbestfitness :
            gbestfitness = pbestfitness.max()
            gbestpop = pop[pbestfitness.argmax()].copy()

    pop20 = [] # 找出前一半数量的优秀粒子
    fitness_list = fitness.tolist()
    max_num_index=map(fitness_list.index, heapq.nlargest(sizepop//2,fitness_list))
    for index in max_num_index:
        pop20.append(pop[index])
        #result[i] = gbestfitness
    # 返回最优粒子
    return gbestpop, pop20, pop, fitness,gbestfitness
    #plt.plot(result)
    #plt.show()
