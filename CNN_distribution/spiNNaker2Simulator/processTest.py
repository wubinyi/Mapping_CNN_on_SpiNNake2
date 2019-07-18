from multiprocessing import Pool, cpu_count, Process
import time
import os

NUM_PROCESS = 36
class Process_Class():
    #因为Process类本身也有__init__方法，这个子类相当于重写了这个方法，
    #但这样就会带来一个问题，我们并没有完全的初始化一个Process类，所以就不能使用从这个类继承的一些方法和属性，
    #最好的方法就是将继承类本身传递给Process.__init__方法，完成这些初始化操作
    def __init__(self):
        self.counterArray = []
        for index in range(NUM_PROCESS):
            counter = Counter([0,index])
            self.counterArray.append(counter)
        print("cpu_count(): {}".format(cpu_count()))
        self.result = []

    def runCounter(self, counterIndex):
        return self.counterArray[counterIndex].run(None)

    def processCallBack(self, result):
        self.result.append(result)
 
    def run(self, singleProcess, task):
        self.result = []
        if singleProcess:
            for index in range(NUM_PROCESS):
                self.result.append(self.counterArray[index].run(task))
            print("result: {}".format(self.result))
        else:
            # pool = Pool(cpu_count())
            # for index in range(NUM_PROCESS):
            #     pool.apply_async(self.counterArray[index].run, (task, ), callback=self.processCallBack)
            # pool.close()
            # pool.join()
            # print("result: {}".format(self.result))
            

class Counter():
    def __init__(self, id):
        self.id = id
        self.clockCounter = 0

    def run(self, task):
        # print("Counter-%s: 子进程(%s) 开始执行，父进程为（%s）"%(self.id, os.getpid(),os.getppid()))
        print("task : {}".format(task))
        for _ in range(200000):
            self.clockCounter += 1
        # print("self.clockCounter : {}".format(self.clockCounter))
        return "{}-{}".format(self.id, self.clockCounter)

class CounterProcess(Process):
    def __init__(self, id):
        Process.__init__(self)
        self.id = id
        self.clockCounter = 0

    def run(self, task):
        # print("Counter-%s: 子进程(%s) 开始执行，父进程为（%s）"%(self.id, os.getpid(),os.getppid()))
        print("task : {}".format(task))
        for _ in range(200000):
            self.clockCounter += 1
        # print("self.clockCounter : {}".format(self.clockCounter))
        return "{}-{}".format(self.id, self.clockCounter)
 
if __name__=="__main__":

    
    print("当前程序进程(%s)"%os.getpid())        
    p1 = Process_Class()
    looptime = 4

    print("----------singleProcess---------")
    for index in range(looptime):
        t_start = time.time()
        p1.run(singleProcess=True, task=index)
        t_stop = time.time()
        print("(%s)执行结束，耗时%0.2f"%(os.getpid(),t_stop-t_start))

    p2 = Process_Class()
    print("----------multiProcess---------")
    for index in range(looptime):
        t_start = time.time()
        p2.run(singleProcess=False, task=index)
        t_stop = time.time()
        print("(%s)执行结束，耗时%0.2f"%(os.getpid(),t_stop-t_start))
