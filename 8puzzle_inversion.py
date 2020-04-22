# Programming Assignment 3 Extra Credit: 8-puzzle
# author: Ron Chen.8336

import sys
import heapq
import copy
import math

class Node(object):
	def __init__(self, state:list, parent, action:tuple, heur:int, depth:int):
		self.state = state
		self.parent = parent
		self.action = action
		self.heur = heur
		self.depth = depth

	def __repr__(self):
		# Printing root as a special case
		if self.parent != None:
			return "STATE: %r PARENT: %r ACTION: %r HEURISTIC: %d DEPTH: %d" % (self.state, self.parent.state, self.action, self.heur, self.depth)
		else:
			return "STATE: %r PARENT: %r ACTION: %r HEURISTIC: %d DEPTH: %d" % (self.state, self.parent, self.action, self.heur, self.depth)

	# For priority queue comparison: f(x) = g(x) + h(x)
	def __lt__(self, other):
		return (self.heur + self.depth) < (other.heur + other.depth)

class AStarAgent(object):

	def __init__(self, file1_name:str, file2_name:str):
		state = []
		goal = []
		fid1 = open(file1_name, 'r')
		fid2 = open(file2_name, 'r')
		for line in fid1:
			state.extend(line.split())
		for line in fid2:
			goal.extend(line.split())
		fid1.close()
		fid2.close()

		state = [int(i) for i in state]
		self.goal = [int(i) for i in goal]
		heur = 8
		self.root = Node(state, None, None, heur, 0)
	
	@staticmethod
	def successor(state:list) -> dict:
		result = {}
		blank = state.index(0)
		if (blank - 1) // 3 == blank // 3:
			action = state[blank - 1]
			result[action] = copy.deepcopy(state)
			result[action][blank] = action
			result[action][blank - 1] = 0
		if (blank + 1) // 3 == blank // 3:
			action = state[blank + 1]
			result[action] = copy.deepcopy(state)
			result[action][blank] = action
			result[action][blank + 1] = 0
		if blank - 3 >= 0:
			action = state[blank - 3]
			result[action] = copy.deepcopy(state)
			result[action][blank] = action
			result[action][blank - 3] = 0
		if blank + 3 < 9:
			action = state[blank + 3]
			result[action] = copy.deepcopy(state)
			result[action][blank] = action
			result[action][blank + 3] = 0
		return result

	@staticmethod
	def heuristic(state:list, goal:list) -> int:
                inversion = 0
                cost = 0
                for i in range(9):
                        cost += abs(goal.index(state[i]) - i)
                        for j in range(i, 9):
                                if state[i] != 0 and state[j] != 0 and state[i] > state[j]:
                                        inversion += 1

                if cost > 0:
                        return inversion + math.floor(math.log2(cost))
                else:
                        return 0

	def AstarSearch(self) -> Node:
		node_count = 0
		frindge = []
		expanded = set()
		heapq.heappush(frindge, self.root)

		while len(frindge) > 0:
			node = heapq.heappop(frindge)
			if node.heur == 0:
				print("GOAL -", node)
				print("Number of nodes expanded: %d, Total path cost: %d" % (node_count, node.depth))
				print("--- END OF SEARCH ---")
				return node
			else:
				expanded.add(tuple(node.state))
				print(node)
				node_count += 1
				depth = node.depth + 1

				children = AStarAgent.successor(node.state)
				for child in children:
					st = children.get(child)
					# Repeated states can never give us any better solutions
					if tuple(st) not in expanded:
						heur = AStarAgent.heuristic(st, self.goal)
						# Naive implementation of decrease_key()
						for nd in frindge:
							if nd.state == st and (depth + heur) < (nd.depth + nd.heur):
								frindge.remove(nd)
								break
						n = Node(st, node, child, heur, depth)
						heapq.heappush(frindge, n)

def solution(goal:Node) -> None:
	n = goal
	path = []
	while n.parent != None:
		path.insert(0, n.action)
		n = n.parent
	print("Solution: ", str(path))

def main(argv:list):
	file1_name = argv[1]
	file2_name = argv[2]
	agent = AStarAgent(file1_name, file2_name)
	goal_node = agent.AstarSearch()
	if goal_node == None:
		print("Unsolvable.")
	else: 
		solution(goal_node)

main(sys.argv)
