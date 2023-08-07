def reward(rList, cNum):
    if rList[0] <= cNum <= rList[1]:
        rDist = 100
    else:     
        rNear = min(rList, key=lambda x:abs(x-cNum))
        rDist = abs(rNear - cNum) * -0.05
    return rDist 

def totalReward(mList, state):
    total_reward = 0.0
    for i in range(len(mList)):
        total_reward += reward(mList[i],state[i])
    return total_reward    

