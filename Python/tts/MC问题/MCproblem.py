from queue import PriorityQueue

#题设条件
M=int(input("传教士数："))
C=int(input("野人数："))
K=int(input("船的最大载人数：")) 

# 定义状态类
class Node:
    def __init__(self,missionaries, cannibals, boat,depth,parent):
        self.missionaries = missionaries
        self.cannibals = cannibals
        self.boat = boat
        self.depth = depth
        self.parent = parent
        self.state = (self.missionaries, self.cannibals, self.boat)
        self.priority = depth + missionaries + cannibals - K*boat  #设置f(n)
                          
    def is_valid(self):  # 检查状态的合法性
        if self.missionaries < 0 or self.cannibals < 0:
            return False
        if self.missionaries > M or self.cannibals > C:
            return False
        if self.cannibals > self.missionaries > 0:
            return False
        if C - self.cannibals > M - self.missionaries > 0:
            return False
        return True

    def is_goal(self):  # 检查是否达到目标状态
        return self.missionaries == 0 and self.cannibals == 0 and self.boat == 0

    def Child_Nodes(self): # 生成每个节点的子节点表
        Child_Nodes = []
        if self.boat == 1: # 船在左岸
            for m in range(self.missionaries+1):
                for c in range(self.cannibals+1):       #穷举可执行动作
                    if 1 <= m + c <= K:  #判断动作是否合法
                        Child_Node = Node(self.missionaries - m, self.cannibals - c, 0,self.depth+1,self)
                        if Child_Node.is_valid(): #p判断后续状态是否合法
                            Child_Nodes.append(Child_Node) #若合法，加入子节点列表
                            
        else:             # 船在右岸,同理
            for m in range(M-self.missionaries+1):
                for c in range(C-self.cannibals+1):
                    if 1 <= m + c <= K:
                        Child_Node = Node(self.missionaries + m, self.cannibals + c, 1,self.depth+1,self)
                        if Child_Node.is_valid():
                            Child_Nodes.append(Child_Node)
                            
        return Child_Nodes
      
    def __lt__(self, other):
        return self.priority < other.priority
    
    
def solve():   # A* 算法
    node = Node(M, C, 1, 0, None)  # 定义初始状态
    if node.is_goal():             # 是否为目标状态
        return "You are already at the goal!"
    
    else:   
        frontier = PriorityQueue()  #定义待扩展节点队列
        visited = {}             #定义已访问节点字典
        visited[node.state]=node.depth    #将初始节点加入已访问列表
        
        for Child_Node in node.Child_Nodes():                  #放入初始态的子节点
            frontier.put(Child_Node)    

        while not frontier.empty():            #当存在节点可拓展时 
            node = frontier.get()              #选择一个节点拓展
            visited[node.state]=node.depth            #将该节点加入已访问列表

            if node.is_goal():                 #检查当前节点是否为目标状态
                path = []                      #若是，回溯路径 
                path.append(node)
                while node.parent is not None:
                    path.append(node.parent)
                    node = node.parent
                path.reverse()
                return path                    #返回路径
            else:                              #若未达到目标状态，继续拓展
                for Child_Node in node.Child_Nodes(): 
                    if Child_Node.state not in visited:                 #判断子节点状态是否已访问
                        frontier.put(Child_Node)                        #否，直接拓展
                    else:                                               #是，比较深度
                        if Child_Node.depth < visited[Child_Node.state]:   
                            visited[Child_Node.state]=Child_Node.depth  #当前深度更小，则替换
                            frontier.put(Child_Node)                    #拓展该节点
    return None

# 解决问题并打印路径
path = solve()
if path is not None:
    print(f"\nSolution found with {len(path)-1} steps!\n")
    print("{:<10} {:<13} {:<13} {:<10}".format("Step", "Missionaries", "Cannibals", "Boat"))
    print("--------------------------------------------------")
    for i, node in enumerate(path):
        print("{:<10} {:<13} {:<13} {:<10}".format(i, node.missionaries, node.cannibals, ['   | R ', ' L |   '][node.boat]))
        print("--------------------------------------------------")
else:
    print("No solution found.") 