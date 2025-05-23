
states = ['Arad', 'Zerind', 'Oradea', 'Sibiu', 'Timisoara', 
        'Lugoj', 'Mehadia', 'Drobeta', 'Craiova', 'Rimnicu Vilcea', 
        'Fagaras', 'Pitesti', 'Bucharest', 'Giurgiu', 'Urziceni', 
        'Hirsova', 'Eforie', 'Vaslui', 'Iasi', 'Neamt']

actions = {'Arad': ['toZerind', 'toSibiu', 'toTimisoara'],
        'Zerind': ['toArad', 'toOradea'],
        'Oradea': ['toZerind', 'toSibiu'],
        'Sibiu': ['toArad', 'toOradea', 'toFagaras', 'toRimnicu Vilcea'],
        'Timisoara': ['toArad', 'toLugoj'],
        'Lugoj': ['toTimisoara', 'toMehadia'],
        'Mehadia': ['toLugoj', 'toDrobeta'],
        'Drobeta': ['toMehadia', 'toCraiova'],
        'Craiova': ['toDrobeta', 'toRimnicu Vilcea', 'toPitesti'],
        'Rimnicu Vilcea': ['toSibiu', 'toCraiova', 'toPitesti'],
        'Fagaras': ['toSibiu', 'toBucharest'],
        'Pitesti': ['toRimnicu Vilcea', 'toCraiova', 'toBucharest'],
        'Bucharest': ['toFagaras', 'toPitesti', 'toGiurgiu', 'toUrziceni'],
        'Giurgiu': ['toBucharest'],
        'Urziceni': ['toBucharest', 'toHirsova', 'toVaslui'],
        'Hirsova': ['toUrziceni', 'toEforie'],
        'Eforie': ['toHirsova'],
        'Vaslui': ['toUrziceni', 'toIasi'],
        'Iasi': ['toVaslui', 'toNeamt'],
        'Neamt': ['toIasi']}

transition_model = {
    'Arad': {'toZerind': 'Zerind', 'toSibiu': 'Sibiu', 'toTimisoara': 'Timisoara'},
    'Zerind': {'toArad': 'Arad', 'toOradea': 'Oradea'},
    'Oradea': {'toZerind': 'Zerind', 'toSibiu': 'Sibiu'},
    'Sibiu': {'toArad': 'Arad', 'toOradea': 'Oradea', 'toFagaras': 'Fagaras', 'toRimnicu Vilcea': 'Rimnicu Vilcea'},
    'Timisoara': {'toArad': 'Arad', 'toLugoj': 'Lugoj'},
    'Lugoj': {'toTimisoara': 'Timisoara', 'toMehadia': 'Mehadia'},
    'Mehadia': {'toLugoj': 'Lugoj', 'toDrobeta': 'Drobeta'},
    'Drobeta': {'toMehadia': 'Mehadia', 'toCraiova': 'Craiova'},
    'Craiova': {'toDrobeta': 'Drobeta', 'toRimnicu Vilcea': 'Rimnicu Vilcea', 'toPitesti': 'Pitesti'},
    'Rimnicu Vilcea': {'toSibiu': 'Sibiu', 'toCraiova': 'Craiova', 'toPitesti': 'Pitesti'},
    'Fagaras': {'toSibiu': 'Sibiu', 'toBucharest': 'Bucharest'},
    'Pitesti': {'toRimnicu Vilcea': 'Rimnicu Vilcea', 'toCraiova': 'Craiova', 'toBucharest': 'Bucharest'},
    'Bucharest': {'toFagaras': 'Fagaras', 'toPitesti': 'Pitesti', 'toGiurgiu': 'Giurgiu', 'toUrziceni': 'Urziceni'},
    'Giurgiu': {'toBucharest': 'Bucharest'},
    'Urziceni':{'toBucharest': 'Bucharest', 'toHirsova': 'Hirsova', 'toVaslui': 'Vaslui'},
    'Hirsova': {'toUrziceni': 'Urziceni', 'toEforie': 'Eforie'},
    'Eforie': {'toHirsova': 'Hirsova'},
    'Vaslui': {'toUrziceni': 'Urziceni', 'toIasi': 'Iasi'},
    'Iasi': {'toVaslui': 'Vaslui', 'toNeamt': 'Neamt'},
    'Neamt': {'toIasi': 'Iasi'}}


cost = {'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
        'Zerind': {'Arad': 75, 'Oradea': 71},
        'Oradea': {'Zerind': 71, 'Sibiu': 151},
        'Sibiu': {'Arad': 140, 'Oradea': 151, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
        'Timisoara': {'Arad': 118, 'Lugoj': 111},
        'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
        'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
        'Drobeta': {'Mehadia': 75, 'Craiova': 120},
        'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
        'Rimnicu Vilcea': {'Sibiu': 80, 'Craiova': 146, 'Pitesti': 97},
        'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
        'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
        'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
        'Giurgiu': {'Bucharest': 90},
        'Urziceni': {'Bucharest': 85, 'Hirsova': 98, 'Vaslui': 142},
        'Hirsova': {'Urziceni': 98, 'Eforie': 86},
        'Eforie': {'Hirsova': 86},
        'Vaslui': {'Urziceni': 142, 'Iasi': 92},
        'Iasi': {'Vaslui': 92, 'Neamt': 87},
        'Neamt': {'Iasi': 87}}


h_dlr = {'Arad': 366, 'Bucharest': 0, 'Craiova': 160, 'Drobeta': 242, 
         'Eforie': 161, 'Fagaras': 176, 'Giurgiu': 77, 'Hirsova': 151, 
         'Iasi': 226, 'Lugoj': 244, 'Mehadia': 241, 'Neamt': 234, 
         'Oradea': 380, 'Pitesti': 100, 'Rimnicu Vilcea': 193, 
         'Sibiu': 253, 'Timisoara': 329, 'Urziceni': 80, 
         'Vaslui': 199, 'Zerind': 374}

class Node:
    
    def __init__(self, state, parent = None, action = None, path_cost = 0):
        self.state = state          # o estado ao qual o nó corresponde; (str)
        self.parent = parent        # o nó da árvore que gerou este nó; (Node)
        self.action = action        # a ação executada para gerar este nó; (str)
        self.path_cost = path_cost  # o custo do caminho do nó inicial até este nó. (int)

class Frontier:
    def __init__(self):
        self.elements = []          # lista de nós
    def is_empty(self):
        # IS-EMPTY(frontier) returns true only if there are no nodes in the frontier.
        return len(self.elements) == 0
    def pop(self):
        # POP(frontier) removes the top node from the frontier and returns it.
        return self.elements.pop(0)
    def top(self):
        # TOP(frontier) returns (but does not remove) the top node of the frontier.
        return self.elements[0]
    def add(self, node):
        #ADD(node, frontier) inserts node into its proper place in the queue.
        pass

class PriorityQueue(Frontier):
        
    def __init__(self, f):
        super().__init__()            # lista de nós
        self.f = f                  # função de avaliação f
    
    def add(self, node):
        # ADD(node, frontier) inserts node into its proper place in the queue.
        self.elements.append(node)
        self.elements = sorted(self.elements, key = self.f)

class FIFOQueue(Frontier):
        
    def add(self, node):
        '''
        ADD(node, frontier) inserts node at the end of the queue.
        '''
        self.elements.append(node)

class LIFOQueue(Frontier):
            
        def add(self, node):
            '''
            ADD(node, frontier) inserts node at the beginning of the queue.
            '''
            self.elements.insert(0, node)

class PriorityQueueK(Frontier): 
        
    def __init__(self,k, f):
        super().__init__()          # lista de nós
        self.f = f                  # função de avaliação f
        self.k = k                  # número de nós a serem expandidos
    
    def add(self, node):
        # ADD(node, frontier) inserts node into its proper place in the queue.
        self.elements.append(node)
        self.elements = sorted(self.elements, key = self.f)
        if len(self.elements) > self.k:
            self.elements = self.elements[:self.k]

class Problem:

    def __init__(self, states, initial, goal, actions, transition_model, cost):
        self.states = states                #estados possíveis
        if initial not in states:           #verifica se o estado inicial é um estado possível
            self.states.append(initial)     #caso não seja, adiciona o estado inicial aos estados possíveis
        self.initial = initial              #estado inicial do problema
        if goal not in states:              #verifica se o estado objetivo é um estado possível
            self.states.append(goal)        #caso não seja, adiciona o estado objetivo aos estados possíveis
        self.goal = goal                    #estado(s) objetivo do problema
        self.actions = actions              #ações possíveis
        self.transition_model = transition_model
        self.cost = cost
    def get_actions(self, state):
        return self.actions[state]
    def result(self, state, action):
        return self.transition_model[state][action]
    def goal_test(self, state):
        return state == self.goal
    def action_cost(self, state1, action, state2):
        if action in self.actions[state1] and state2 == self.result(state1, action):
            return self.cost[state1][state2]
        else:
            return -1
        
'''
function BEST-FIRST-SEARCH(problem, f ) returns a solution node or failure 
    node←NODE(STATE=problem.INITIAL)
    frontier ← a priority queue ordered by f , with node as an element
    reached←a lookup table, with one entry with key problem.INITIAL and value node 
    while not IS-EMPTY(frontier) do
        node←POP(frontier)
        if problem.IS-GOAL(node.STATE) then return node 
        for each child in EXPAND(problem, node) do
            s←child.STATE
            if s is not in reached or child.PATH-COST < reached[s].PATH-COST then
                reached[s] ← child
                add child to frontier 
    return failure

function EXPAND(problem,node) yields nodes 
    s←node.STATE
    for each action in problem.ACTIONS(s) do
        s′ ←problem.RESULT(s,action)
        cost←node.PATH-COST + problem.ACTION-COST(s,action,s′)
        yield NODE(STATE=s′, PARENT=node, ACTION=action, PATH-COST=cost)
'''
class BestFirstSearch:
    def __init__(self, problem, f):
        self.problem = problem
        # frontier ← a priority queue ordered by f , with node as an element
        self.reached = {} # a lookup table
        self.frontier = PriorityQueue(f) 
        self.f = f

    def search(self, limit = 20):
        nExpanded = 0
        node = Node(self.problem.initial) # node←NODE(STATE=problem.INITIAL)
        self.frontier.add(node) # add node to frontier
        self.reached = {self.problem.initial: node} # reached←a lookup table, with one entry with key problem.INITIAL and value node
        while not self.frontier.is_empty(): # while not IS-EMPTY(frontier) do
            print("Frontier: ", end="")
            for node in self.frontier.elements:
                print(node.state, end=" ")
            print()
            node = self.frontier.pop() # node←POP(frontier)
            nExpanded += 1
            if nExpanded > limit:
                return None
            if self.problem.goal_test(node.state):
                return node
            else:
                print(f"{nExpanded} - {node.state} {node.path_cost} {self.f(node)}")
            for child in self.expand(node):
                s = child.state
                if s not in self.reached or child.path_cost < self.reached[s].path_cost:
                    self.reached[s] = child
                    self.frontier.add(child)
        return None
    
    def expand(self, node):
        s = node.state
        for action in self.problem.get_actions(s):
            s_prime = self.problem.result(s, action)
            if node.parent and s_prime == node.parent.state:
                #print(f"Skipping {s_prime} because it is the parent of {node.state}")
                continue
            cost = node.path_cost + self.problem.action_cost(s, action, s_prime)
            yield Node(s_prime, node, action, cost)

    def path(self, node):
        path_back = []
        while node:
            path_back.append(node)
            node = node.parent
        return path_back[::-1]


# for Breadth-first search as a BestFirstSearch
def depth(node):
    node_depth = 0
    while node:
        node_depth += 1
        node = node.parent
    return node_depth

def n_depth(node):
    return -1*depth(node)

# Breadth-first search as a BestFirstSearch with FIFOQueue
# as frontier 
class BreadthFirstSearch(BestFirstSearch):

    def __init__(self, problem):
        super().__init__(problem, None)
        #self.problem = problem
        # frontier ← as a FIFO queue
        self.frontier = FIFOQueue() 
        #self.reached = {} # a lookup table, with one entry with key problem.INITIAL and value node
    
    def f(self, node):
        node_depth = 0
        while node:
            node_depth += 1
            node = node.parent
        return node_depth

class DepthFirstSearch(BestFirstSearch):
    def __init__(self, problem):
        self.problem = problem        
        self.frontier = PriorityQueue(self.f) 

    def search(self, limit = 20):
        nExpanded = 0
        node = Node(self.problem.initial) # node←NODE(STATE=problem.INITIAL)
        self.frontier.add(node) # add node to frontier
        while not self.frontier.is_empty(): # while not IS-EMPTY(frontier) do
            print("Frontier: ", end="")
            for node in self.frontier.elements:
                print(node.state, end=" ")
            print()
            node = self.frontier.pop() # node←POP(frontier)
            nExpanded += 1
            if nExpanded > limit:
                return None
            if self.problem.goal_test(node.state):
                return node
            else:
                print(f"{nExpanded} - {node.state} {node.path_cost} {self.f(node)}")
            for child in self.expand(node):
                self.frontier.add(child)
    def f(self, node):
        node_depth = 0
        while node:
            node_depth += 1
            node = node.parent
        return -1*node_depth

class DepthFirstSearchNoCycles(DepthFirstSearch):
    def search(self, limit = 20):
        nExpanded = 0
        node = Node(self.problem.initial) # node←NODE(STATE=problem.INITIAL)
        self.frontier.add(node) # add node to frontier
        while not self.frontier.is_empty(): # while not IS-EMPTY(frontier) do
            print("Frontier: ", end="")
            for node in self.frontier.elements:
                print(node.state, end=" ")
            print()
            node = self.frontier.pop() # node←POP(frontier)
            nExpanded += 1
            if nExpanded > limit:
                return None
            if self.problem.goal_test(node.state):
                return node
            else:
                print(f"{nExpanded} - {node.state} {node.path_cost} {self.f(node)}")
            for child in self.expand(node):
                if self.isCycle(child):
                    continue
                self.frontier.add(child)

    def isCycle(self, node):
        # check if the node is a cycle
        s = node.state
        while node.parent:
            #print(f"Checking {node.parent.state} against {s}")
            if node.parent.state == s:
                return True
            node = node.parent
        return False


""" 
function DEPTH-LIMITED-SEARCH(problem, l) returns a node or failure or cutoff 
    frontier←a LIFO queue (stack) with NODE(problem.INITIAL) as an element 
    result ← failure
    while not IS-EMPTY(frontier) do
        node←POP(frontier)
        if problem.IS-GOAL(node.STATE) then return node 
        if DEPTH(node) > l then
            result ← cutoff
        else if not IS-CYCLE(node) do
            for each child in EXPAND(problem, node) do 
                add child to frontier
    return result
"""

class DepthLimitedSearch(DepthFirstSearchNoCycles):

    def __init__(self, problem):
        self.problem = problem        
        self.frontier = LIFOQueue()

    def search(self, l):
        node = Node(self.problem.initial)
        self.frontier.add(node)
        result = None
        nExpanded = 0
        while not self.frontier.is_empty():
            node = self.frontier.pop()
            if self.problem.goal_test(node.state):
                return node
            else:
                nExpanded += 1
                print(f"{nExpanded} - {node.state} {node.path_cost} {self.depth(node)}")
            if self.depth(node) >= l:
                result = 'cutoff'
            elif not self.isCycle(node):
                for child in self.expand(node):
                    self.frontier.add(child)
        return result

    def depth(self, node):
        depth = 0
        while node:
            depth += 1
            node = node.parent
        return depth
    

class IterativeDeepeningSearch(DepthLimitedSearch):

    def __init__(self, problem):
        super().__init__(problem)
        

    def search(self):
        for depth in range(0, 10000):
            print()
            print(f"Depth: {depth}")
            print("__________________________")
            result = super().search(depth)
            if result != 'cutoff':
                return result
        return None
    

"""
function BIBF-SEARCH(problemF, fF, problemB, fB) returns a solution node, or failure 
    nodeF ← NODE(problemF .INITIAL) // Node for a start state
    nodeB ← NODE(problemB.INITIAL) // Node for a goal state 
    frontierF ← a priority queue ordered by fF, with nodeF as an element
    frontierB ← a priority queue ordered by fB, with nodeB as an element 
    reachedF ← a lookup table, with one key nodeF.STATE and value nodeF 
    reachedB ← a lookup table, with one key nodeB.STATE and value nodeB 
    solution ← failure
    while not TERMINATED(solution, frontierF , frontierB) do 
        if fF (TOP(frontierF )) < fB(TOP(frontierB)) then
            solution←PROCEED(F, problemF, frontierF, reachedF, reachedB, solution) 
        else solution←PROCEED(B, problemB, frontierB, reachedB, reachedF, solution)
    return solution

function PROCEED(dir, problem, frontier, reached, reached2, solution) returns a solution 
        // Expand node on frontier; check against the other frontier in reached2.
        // The variable “dir” is the direction: either F for forward or B for backward.
    node←POP(frontier)
    for each child in EXPAND(problem, node) do
        s←child.STATE
        if s not in reached or PATH-COST(child) < PATH-COST(reached[s]) then
            reached[s] ← child
            add child to frontier
            if s is in reached2 then
                solution2 ← JOIN-NODES(dir, child, reached2[s]))
                if PATH-COST(solution2) < PATH-COST(solution) then
                    solution ← solution2 
    return solution
"""
class BidirectionalBestFirstSearch:
    def __init__(self, problemF, fF, problemB, fB):
        self.problemF = problemF
        self.fF = fF
        self.problemB = problemB
        self.fB = fB
        self.frontierF = PriorityQueue(fF)
        self.frontierB = PriorityQueue(fB)
        self.reachedF = {}
        self.reachedB = {}

    def search(self):
        nodeF = Node(self.problemF.initial)
        nodeB = Node(self.problemB.initial)
        self.frontierF.add(nodeF)
        self.frontierB.add(nodeB)
        self.reachedF = {self.problemF.initial: nodeF}
        self.reachedB = {self.problemB.initial: nodeB}
        solution = None
        while not self.terminated(solution, self.frontierF, self.frontierB):
            if self.fF(self.frontierF.top()) < self.fB(self.frontierB.top()):
                print("Expanding forward")
                solution = self.proceed('F', self.frontierF, self.reachedF, 
                                        self.reachedB, solution)
                if solution is not None:
                    print(f"Solution found: {solution[0].state} {solution[1].state}")
            else:
                print("Expanding backward")
                solution = self.proceed('B', self.frontierB, self.reachedB, 
                                        self.reachedF, solution)
                if solution is not None:
                    print(f"Solution found: {solution[0].state} {solution[1].state}")
        return solution
    
    def proceed(self, direction, frontier, reached, reached2, solution):
        node = frontier.pop()
        print(f"{node.state} {node.path_cost} {self.depth(node)}")
        for child in self.expand(node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                frontier.add(child)
                if s in reached2:
                    solution2 = self.join_nodes(direction, child, reached2[s])
                    if solution is None or solution2[0].path_cost < solution[0].path_cost:
                        solution = solution2

        return solution
    
    def join_nodes(self, direction, node1, node2):
        if direction == 'F':
            return node1, node2
        else:
            return node2, node1
        
    def expand(self, node):
        s = node.state
        for action in self.problemF.get_actions(s):
            s_prime = self.problemF.result(s, action)
            cost = node.path_cost + self.problemF.action_cost(s, action, s_prime)
            yield Node(s_prime, node, action, cost)

    def terminated(self, solution, frontierF, frontierB):
        return solution is not None or frontierF.is_empty() or frontierB.is_empty()
    
    def path(self, node):
        path_back = []
        nodeF = node[0]
        nodeB = node[1].parent
        while nodeF:
            path_back.append(nodeF)
            nodeF = nodeF.parent
    
        path_back = path_back[::-1]
        while nodeB:
            path_back.append(nodeB)
            nodeB = nodeB.parent
        return path_back
    
    
    def depth(self, node):
        depth = 0
        while node:
            depth += 1
            node = node.parent
        return depth





def main():
    
    initial = 'Arad'
    goal = 'Bucharest'
    
    Arad2Bucarest = Problem(states, initial, goal, actions, transition_model, cost)

    print("1. Uninformed Search Strategies")
    print("1.1. Breadth-First Search")
    print(" * This is a systematic search strategy that is therefore ")
    print("complete even on infinite state spaces.")
    print(" * We could implement breadth-first search as a call to ")
    print("BEST-FIRST-SEARCH where the evaluation function $f(n)$")
    print("is the depth of the node—that is, the number of actions ")
    print("it takes to reach the node.")

    bfs_depth = BestFirstSearch(Arad2Bucarest, depth)

    print("Breadth-first search (Arad -> Bucharest):")
    node = bfs_depth.search()
    print(node.state, node.path_cost)
    print("__________________________")
    print("Solution:")
    for step in bfs_depth.path(node):
        print(step.state, step.path_cost)
    print("__________________________")

    print(" * A first-in-first-out queue will be faster than a ")
    print("priority queue, and will give us the correct order of") 
    print("nodes: new nodes (which are always deeper than their ")
    print("parents) go to the back of the queue, and old nodes, ")
    print("which are shallower than the new nodes, get expanded ")
    print("first. ")

    bfs_FIFO = BreadthFirstSearch(Arad2Bucarest)
    print("Breadth-first search (Arad -> Bucharest):")
    node = bfs_FIFO.search()
    print(node.state, node.path_cost)
    print("__________________________")
    print("Solution:")
    for step in bfs_FIFO.path(node):
        print(step.state, step.path_cost)
    print("__________________________")

    print("1.2. Depth-First Search")

    dfs_depth = BestFirstSearch(Arad2Bucarest, n_depth)
    print("Depth-first search (Arad -> Bucharest):")
    node = dfs_depth.search(limit=10)
    if node is None:
        print("No solution found")
    else:
        print(node.state, node.path_cost)
        print("__________________________")
        print("Solution:")
        for step in dfs_depth.path(node):
            print(step.state, step.path_cost)
    print("__________________________")



if __name__ == "__main__":
    main()


