from Scripts.Algorithm import train, evaluateMFC, evaluateMARL
from Scripts.Parameters import ParseInput
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

if __name__ == '__main__':
    args = ParseInput()

    if not os.path.exists('Results'):
        os.mkdir('Results')

    # Logging
    args.logFileName = 'Results/progress.log'
    open(args.logFileName, 'w').close()
    logging.basicConfig(filename=args.logFileName,
                        format='%(asctime)s %(message)s',
                        filemode='w')
    args.logger = logging.getLogger()
    args.logger.setLevel(logging.INFO)

    t0 = time.time()

    indexN = 0
    valueRewardMFCArray = np.zeros(args.numN)
    valueRewardMFCArraySD = np.zeros(args.numN)

    valueRewardMARLArray = np.zeros(args.numN)
    valueRewardMARLArraySD = np.zeros(args.numN)

    percentRewardErrorArray = np.zeros(args.numN)
    percentRewardErrorArraySD = np.zeros(args.numN)

    NVec = np.zeros(args.numN)

    if args.train:
        args.logger.info('Training is in progress.')
        train(args)

    args.logger.info('Evaluation is in progress.')
    while indexN < args.numN:
        N = args.minN + indexN * args.divN
        NVec[indexN] = N

        for _ in range(0, args.maxSeed):
            valueRewardMFC = evaluateMFC(args)
            valueRewardMFC = np.array(valueRewardMFC.detach())

            valueRewardMFCArray[indexN] += valueRewardMFC/args.maxSeed
            valueRewardMFCArraySD[indexN] += valueRewardMFC ** 2 / args.maxSeed

            valueRewardMARL = evaluateMARL(args, N)
            valueRewardMARL = np.array(valueRewardMARL.detach())

            valueRewardMARLArray[indexN] += valueRewardMARL/args.maxSeed
            valueRewardMARLArraySD[indexN] += valueRewardMARL**2/args.maxSeed

            percentRewardError = np.abs((valueRewardMARL - valueRewardMFC)/valueRewardMFC) * 100
            percentRewardErrorArray[indexN] += percentRewardError/args.maxSeed
            percentRewardErrorArraySD[indexN] += percentRewardError**2/args.maxSeed

        indexN += 1
        args.logger.info(f'Evaluation N: {N}')

    valueRewardMFCArraySD = np.sqrt(np.maximum(0, valueRewardMFCArraySD - valueRewardMFCArray ** 2))
    valueRewardMARLArraySD = np.sqrt(np.maximum(0, valueRewardMARLArraySD - valueRewardMARLArray ** 2))
    percentRewardErrorArraySD = np.sqrt(np.maximum(0, percentRewardErrorArraySD - percentRewardErrorArray ** 2))

    plt.figure()
    plt.xlabel('Number of Agents')
    plt.ylabel('Reward Values')
    plt.plot(NVec, valueRewardMFCArray, label='MFC')
    plt.fill_between(NVec, valueRewardMFCArray - valueRewardMFCArraySD, valueRewardMFCArray + valueRewardMFCArraySD, alpha=0.3)
    plt.plot(NVec, valueRewardMARLArray, label='MARL')
    plt.fill_between(NVec, valueRewardMARLArray - valueRewardMARLArraySD, valueRewardMARLArray + valueRewardMARLArraySD, alpha=0.3)
    plt.legend()
    plt.savefig(f'Results/RewardValues{args.sigma}.png')

    plt.figure()
    plt.xlabel('Number of Agents')
    plt.ylabel('Percentage Error')
    plt.plot(NVec, percentRewardErrorArray)
    plt.fill_between(NVec, percentRewardErrorArray - percentRewardErrorArraySD, percentRewardErrorArray + percentRewardErrorArraySD, alpha=0.3)
    plt.savefig(f'Results/RewardError{args.sigma}.png')

    t1 = time.time()

    args.logger.info(f'Elapsed time is {t1-t0} sec')
