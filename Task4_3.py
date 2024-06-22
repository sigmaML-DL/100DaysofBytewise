# Breadth First Search 
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
visited = []
queue = []

def bfs(visited, graph, node):
    visited.append(node)
    queue.append(node)
    traversal_order = []
    
    while queue:
        m = queue.pop(0)
        traversal_order.append(m)
        
        for neighbour in graph[m]:
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)
    
    return traversal_order

result = bfs(visited, graph, 2)
print("BFS Traversal Order starting from node 2:")
print(result)

# Depth First search 
graph = {0: [1, 2], 1: [2], 2: [0, 3], 3: [3]}
visited = []

def dfs(visited, graph, node):
    if node not in visited:
        print(node, end=" ")
        visited.append(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

print("DFS Traversal starting from node 2:")
dfs(visited, graph, 2)


