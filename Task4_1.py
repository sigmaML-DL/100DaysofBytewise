# Question 1
def quick_sort(sequence):
    length=len(sequence)
    if length<= 1:
        return sequence
    else:
        pivot=sequence.pop() # pops the right most element and returns it 
    
    items_greater=[]   
    items_smaller=[] 
    
    for item in sequence:
        if item > pivot:
            items_greater.append(item)
        else:
            items_smaller.append(item)
    
    return   quick_sort(items_smaller) + [pivot]+ quick_sort(items_greater )

print(quick_sort([0,5,2,4,3,1,6]))