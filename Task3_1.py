# Question 1 
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)


number = 67
print(f"The Answer  is: {factorial(number)}")

# Question 2
# Linked Lists 

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def isPalindrome(head):
    if not head or not head.next:
        return True
    
    
    def reverseList(head):
        prev = None
        current = head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev
    
    
    slow = head
    fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next
    
    
    second_half_start = reverseList(slow.next)
    
    
    first_half = head
    second_half = second_half_start
    is_palindrome = True
    while second_half and is_palindrome:
        if first_half.val != second_half.val:
            is_palindrome = False
        first_half = first_half.next
        second_half = second_half.next
    
    
    reverseList(second_half_start)
    
    return is_palindrome

def createLinkedList(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


def printPalindromeResult(values):
    head = createLinkedList(values)
    if isPalindrome(head):
        print("The linked list is a palindrome.")
    else:
        print("The linked list is not a palindrome.")

printPalindromeResult([1, 2, 3, 2, 1]) 


# Question 3   
def merge_sorted_arrays(arr1, arr2):
    
    i, j = 0, 0
    result = []

   
    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    
    while i < len(arr1):
        result.append(arr1[i])
        i += 1

    
    while j < len(arr2):
        result.append(arr2[j])
        j += 1

    return result


arr1 = [1, 3, 5, 7]
arr2 = [2, 4, 6, 8]
merged_array = merge_sorted_arrays(arr1, arr2)
print(merged_array)  

# Question 4
class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        self.root = self._insert_recursive(self.root, key)

    def _insert_recursive(self, root, key):
        if root is None:
            return TreeNode(key)
        
        if key < root.key:
            root.left = self._insert_recursive(root.left, key)
        else:
            root.right = self._insert_recursive(root.right, key)
        
        return root

    def delete(self, key):
        self.root = self._delete_recursive(self.root, key)

    def _delete_recursive(self, root, key):
        if root is None:
            return root
        
        if key < root.key:
            root.left = self._delete_recursive(root.left, key)
        elif key > root.key:
            root.right = self._delete_recursive(root.right, key)
        else:
            
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            
            
            successor = self._find_min(root.right)
            root.key = successor.key
            root.right = self._delete_recursive(root.right, successor.key)
        
        return root
    
    def _find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current
    
    def search(self, key):
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, root, key):
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self._search_recursive(root.left, key)
        else:
            return self._search_recursive(root.right, key)

    def inorder_traversal(self):
        self._inorder_recursive(self.root)
        print()

    def _inorder_recursive(self, root):
        if root:
            self._inorder_recursive(root.left)
            print(root.key, end=' ')
            self._inorder_recursive(root.right)


bst = BST()
bst.insert(50)
bst.insert(30)
bst.insert(20)
bst.insert(40)
bst.insert(70)
bst.insert(60)
bst.insert(80)

print("Inorder traversal of the BST:")
bst.inorder_traversal()  

bst.delete(20)
print("Inorder traversal after deleting 20:")
bst.inorder_traversal()  

bst.delete(30)
print("Inorder traversal after deleting 30:")
bst.inorder_traversal()  

print("Search for key 50:")
result = bst.search(50)
if result:
    print("Key 50 found in the BST.")
else:
    print("Key 50 not found in the BST.")

# Question 5
def longest_palindromic_substring(s):
    n = len(s)
    if n == 0:
        return ""
    
    
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_len = 1
    
   
    for i in range(n):
        dp[i][i] = True
    
    
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_len:
                    max_len = length
                    start = i
    
    return s[start:start + max_len]


input_str = "babad"
print("Answer : ", longest_palindromic_substring(input_str))

# Question 6
def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    
    merged = []
    merged.append(intervals[0])
    
    for current in intervals[1:]:
        
        current_start, current_end = current
       
        last_start, last_end = merged[-1]
        
        if current_start <= last_end:  # Overlapping intervals
            merged[-1] = (last_start, max(last_end, current_end))
        else:  
            merged.append((current_start, current_end))
    
    return merged


i = [(1, 3), (2, 6), (8, 10), (15, 18)]
merged_intervals = merge_intervals(i)
print("Merged intervals:", merged_intervals)

# Question 7 
def max_subarray_sum(nums):
    if not nums:
        return 0
    
    max_sum = nums[0]
    current_sum = nums[0]
    
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print("Answer : ", max_subarray_sum(nums))

# Question 10
# Board game 
def find_words(board, words):
    if not board or not words:
        return []

    m, n = len(board), len(board[0])
    result = set()

    def backtrack(r, c, path, visited):
        if r < 0 or r >= m or c < 0 or c >= n or visited[r][c]:
            return
        path += board[r][c]
        visited[r][c] = True

        if path in words:
            result.add(path)

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            backtrack(r + dr, c + dc, path, visited)

        visited[r][c] = False

    for i in range(m):
        for j in range(n):
            visited = [[False] * n for _ in range(m)]
            backtrack(i, j, "", visited)

    return list(result)

b = [
    ['o', 'a', 'z', 'n'],
    ['e', 't', 'x', 'e'],
    ['i', 'h', 'k', 'r'],
    ['i', 'f', 'l', 'v']
]
w = ["oath", "pea", "eat", "rain"]
print(find_words(b, w))



