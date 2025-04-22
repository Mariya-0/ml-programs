print("bubble sort")

def bubblesort(array, size):
    for i in range(size):
        swapped = False
        for j in range(size - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                swapped = True
        if not swapped:
            break  # If no elements were swapped, the array is already sorted

arr = [-2, 45, 0, 11, -9, 88, -97, -202, 747]
size = len(arr)
bubblesort(arr, size)

print('The array after sorting (bubble sort) in Ascending order is:')
print(arr)
