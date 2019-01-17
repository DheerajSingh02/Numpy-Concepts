
# coding: utf-8

# <h2>NumPy Basics</h2>

# In[4]:


import numpy as np            #Import numpy as np namespace.


# In[5]:


print (np.__version__)        #Check numpy version. 


# <h2>Create NumPy arrays using Python's "array like" data types</h2>

# In[3]:


#In NumPy arrays, all elements are of same data type.NumPy auto-detects the data-type from the input.
# Create NumPy arrays from Python's List

my_list = [-17, 0, 4, 5, 9]
my_array_from_list = np.array(my_list)
my_array_from_list


# In[44]:


my_array_from_list = np.array([1,2,3,4,5])
my_array_from_tuple = np.array((1,2,3,4,5))

print(my_array_from_list)
print(my_array_from_tuple)      #Observe that both arrays which are created from list and tuple are same.


# In[4]:


my_array = np.array([1,2,3,"String_value", True]) 

my_array     #Here in o/p, all elements have been converted to final element dtype -> dtype='<U12' that is NumPy unicode.


# In[5]:


#Note that any arithmatic opertaion performed on the NumPy array is elementwise. E.g,
my_array_from_list * 10             #Here every element gets multiplied by 10.


# In[6]:


#If we do such operation on Python's list, it will be applicable to whole list. E.g,

my_list = [-17, 0, 4, 5, 9]

my_list*2                   #List elements are getting repeated.


# In[7]:


# Create NumPy arrays from Python's tuple. Note that each element in tuple is promoted to the final type within the array after 
# creating the NumPy array.
my_tuple = (14, -3.54, 5+7j)
np.array(my_tuple)


# <h2>Intrinsic NumPy array creation using NumPy's methods</h2>

# In[8]:


# arange() function to create 1-D array. arange(start,stop,step) -> start is inclusive and stop is exclusive.

my_1D_array = np.arange(7)

my_1D_array


# In[9]:


my_1D_array = np.arange(1,10,1.3)
my_1D_array


# In[77]:


print("Dimension is:",my_1D_array.ndim)       # to check dimension of array created
print("Shape is:", my_1D_array.shape)         # to check shape of the array.shape (7,) means the array 
                                              # is indexed by a single index which runs from 0 to 6. 
print("Size is:", my_1D_array.size)           # to check size of the array, it is equal to total no. of elements(rowsXcolumns)
print("Length is:", len(my_1D_array))


# In[11]:


np.arange(10, 26, 5)


# In[12]:


np.arange(26, step=5)


# In[13]:


np.arange(0, 26, step=5)


# In[14]:


np.arange(0, 26, 5)


# <h1>linspace(), zeros(), ones(), and NumPy data types</h1>

# In[16]:


#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#Returns num evenly spaced samples, calculated over the interval [start, stop].
np.linspace(5, 15, 9)


# In[17]:


my_linspace = np.linspace(5, 15, 9, retstep=True)       #step is the spacing between samples.
my_linspace[1]


# In[18]:


np.linspace(5, 15, 9, retstep=True)[1]


# <h2>zeros()</h2>

# In[20]:


#numpy.zeros(shape, dtype=float, order='C') 
#Return a new array of given shape and type, filled with zeros.

np.ones(7)


# In[79]:


np.zeros((5,4))


# In[22]:


np.zeros((5,4,3))


# In[72]:


c = np.eye(3)     #Identity array
c


# In[73]:


d = np.diag(np.array([1, 2, 3, 4])) #Diagonal array
d


# In[74]:


a = np.random.rand(4)       # uniform in [0, 1]
a


# In[75]:


b = np.random.randn(4)      # Gaussian
b


# <h2>NumPy data types()</h2>

# In[23]:


np.zeros(11, dtype='int64')


# In[24]:


np.zeros(11)          # Default dtype is float.


# <h1>Slicing Arrays</h1>

# In[ ]:


#A slicing operation creates a view on the original array, which is just a way of accessing array data. 
#Thus the original array is not copied in memory.Use np.may_share_memory() to check if two arrays share the same memory block.
#Note:When modifying the view, the original array is modified as well.


# In[26]:


my_vector = np.array ([-17, -4, 0, 2, 21, 37, 105])
my_vector


# In[27]:


# Index is zero based.

my_vector[0] 


# In[28]:


my_vector[-3]   # access elements from the last. my_vector[-1] will return the last element ans so on.


# In[29]:


#Index must me within range of total number of elements. otherwise, accessing that element will give an error.
my_vector[305]


# In[31]:


my_vector[305 % 7]   # 305 % 7 = 4 , so it will return the 4th element from array.


# In[32]:


my_vector.size


# In[33]:


my_vector[305 % my_vector.size]     #These are different ways to access elements.


# <h2>Two Dimensional Arrays</h2>

# In[34]:


my_array= np.arange(35)
my_array.shape = (7,5)             # Note that while specifing the shape,the product of shape numbers must be equal to the total no of elements while creating the array.
                                   # Here 7x5 = 35. Here 7 denotes no. of rows and 5 denotes no. of column.
my_array


# In[35]:


#In numpy arrays, dimensionality refers to the number of axes needed to index it, 
#not the dimensionality of any geometrical space
#Rank is number of axes.
#axix 0 refers to rows and axis 1 refers columns.


# In[36]:


my_array.ndim       # to check no of dimensions.


# In[41]:


print(my_array[0])                         #It will print first row.       
print(my_array[-2])                        #It will print second last row
print(my_array[:,0])                       #It will print first column.


# In[42]:


# To select a particular element,specify the value in terms of row and column value.
my_array[5,2]    #It will print 3rd element of 5th row.


# In[43]:


#This example is similar to below one but this is frequently used.
row = 5
column = 2
my_array[row, column]


# In[45]:


my_array[5][2]  #It will also fetch the same value as above method does.


# <h2>Three Dimensional Arrays</h2>

# In[47]:


my_3D_array = np.arange(70)
my_3D_array.shape = (2, 7, 5)
my_3D_array                        #It will create 2 arrays of shape (7,5). 


# In[49]:


my_3D_array[1]      #It will select second array.


# In[51]:


my_3D_array[1,3]          #It will select 4th row from second array.          


# In[53]:


#Note : When modifying the view, the original array is modified as well:
my_3D_array[1,3,2] =1111
my_3D_array


# In[83]:


a = np.arange(10)


# In[84]:


c = a[::2].copy()  # force a copy


# In[85]:


c[0] = 12

print("Array c is:",c)
print("Array a is:",a)


# In[87]:


np.may_share_memory(a, c)


# In[70]:


from IPython.display import Image
Image("path_to_image\\numpy_array.jpg")


# In[68]:


from IPython.display import Image
Image("path_to_image\\numpy_indexing.png")


# <h1>Boolean Mask Arrays</h1>

# In[ ]:


#NumPy arrays can be indexed with slices, but also with boolean or integer arrays (masks). 
#This method is called fancy indexing. It creates copies not views. 


# In[81]:


a = np.random.randint(0, 21, 15)
a


# In[82]:


(a % 3 == 0)  #Check the elements which are divisible by 3


# In[88]:


mask = (a % 3 == 0)
extract_from_a = a[mask] # or,  a[a%3==0]
extract_from_a           # extract a sub-array with the mask


# In[90]:


#Indexing with a mask can be very useful to assign a new value to a sub-array:
a[a % 3 == 0] = -1
a


# <h2>Indexing with an array of integers</h2>

# In[92]:


a = np.arange(0, 100, 10)
a


# In[93]:


a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list


# In[95]:


#New values can be assigned with this kind of indexing:
a[[9, 7]] = -100
a


# In[96]:


#When a new array is created by indexing with an array of integers, the new array has the same shape as the array of integers:
a = np.arange(10)
a


# In[103]:


idx = np.array([[3, 4], [9, 7]])
print("Array idx is:",idx)
print("Shape of idx is:", idx.shape)


# In[101]:


print(a[idx])


# In[104]:


print(a[idx].shape)


# <h2>Few more examples on fancy indexing</h2>

# In[116]:


a = np.arange(6) + np.arange(0,51,10)[:, np.newaxis]     #np.newasix is used to increase the dimension of the existing array by  
                                                         #one more dimension, when used once.
a


# In[110]:


a[(0,1,2,3,4),(1,2,3,4,5)]


# In[111]:


mask = np.array([1,0,1,0,0,1], dtype=bool)
mask


# In[112]:


a[mask] #It will fetch rows corresponding to 'true' i.e row 0, row 2 and row 6


# In[117]:


a[mask,2]    #It will fetch 3rd column.


# In[118]:


a[3:]       #all rows after 4th row(including)


# In[119]:


a[:, [0,2,5]]   #all rows of 1st, 3rd and 6th column i.e equivalent to col 0, col 2 and col 5


# In[120]:


a[3:, [0,2,5]]   #combining above two examples.


# In[121]:


my_vector = np.array([-17, -4, 0, 2, 21, 37, 105])
my_vector


# In[122]:


zero_mod_7_mask = 0 == (my_vector % 7)
zero_mod_7_mask


# In[123]:


sub_array = my_vector[zero_mod_7_mask]
sub_array


# In[124]:


sub_array[sub_array>0]


# In[125]:


mod_test = 0 == (my_vector % 7)
mod_test


# In[126]:


positive_test = my_vector > 0
positive_test


# In[127]:


combined_mask = np.logical_and(mod_test, positive_test)


# In[128]:


my_vector[combined_mask]


# <h2>Broadcasting</h2>

# <h6>The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is “broadcast” across the larger array so that they have compatible shapes.</h6>

# In[134]:


from IPython.display import Image
Image("path_to_image\\broadcasting.png") #The light boxes represent the broadcasted values.


# In[131]:


a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 2.0, 2.0])
a*b                            #element wise multiplication happens.


# In[132]:


#Now let's observe what happens when b is a scalar instead of a vector.
a = np.array([1.0, 2.0, 3.0])
b = 2.0
a * b                #we get the same result as in previous cell. Here b is being stretched during the arithmetic operation 
                     #into an array with the same shape as a


# Note : The new elements in b are simply copies of the original scalar. The stretching analogy is only conceptual.

# Broadcasting rule:
# Rule 1: If the two arrays differ in their number of dimensions, the shape of the one with fewer dimensions is padded with ones on its leading (left) side.
# Rule 2: If the shape of the two arrays does not match in any dimension, the array with shape equal to 1 in that dimension is stretched to match the other shape.
# Rule 3: If in any dimension the sizes disagree and neither is equal to 1, an error is raised.

# In[6]:


#Broadcasting ex 1

M = np.ones((2, 3))               # 2 rows, 3 columns, all values are 1.
a = np.arange(3)
print("Shape of M:", M.shape)
print("Shape of a:", a.shape)


# Here array a has fewer dimensions, so we pad it on the left with ones:
# 
#     M.shape -> (2, 3)
#     a.shape -> (1, 3)
# 
# By rule 2, we now see that the first dimension disagrees, so we stretch this dimension to match:
# 
#     M.shape -> (2, 3)
#     a.shape -> (2, 3)

# In[7]:


M + a


# In[9]:


#Broadcasting ex 2
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print("Shape of a:", a.shape)
print("Shape of b:",b.shape)


# Rule 1 says we must pad the shape of b with ones:
# 
#     a.shape -> (3, 1)
#     b.shape -> (1, 3)
# And rule 2 tells us that we upgrade each of these ones to match the corresponding size of the other array:
# 
#     a.shape -> (3, 3)
#     b.shape -> (3, 3)

# In[10]:


a+b


# In[11]:


#Broadcasting ex 3
M = np.ones((3, 2))
a = np.arange(3)
print("Shape of M:",M.shape)
print("Shape of a:", a.shape)


# Again, rule 1 tells us that we must pad the shape of a with ones:
# 
#     M.shape -> (3, 2)
#     a.shape -> (1, 3)
# 
# By rule 2, the first dimension of a is stretched to match that of M:
# 
#     M.shape -> (3, 2)
#     a.shape -> (3, 3)
#     
# Now we hit rule 3–the final shapes do not match, so these two arrays are incompatible, as we can observe by attempting this operation:

# In[12]:


M+a


# <h2>"np.newaxis"->adds a new axis. i.e, if array is 1-D then it will convert to 2-D and so on.. </h2>

# In[15]:


print("Shape of a befor adding np.newaxis:", a.shape)
print("Shape of a after adding np.newaxis:", a[:, np.newaxis].shape)


# <h2>Basic reductions -> Computing sums </h2>

# In[17]:


x = np.array([1, 2, 3, 4])
print(np.sum(x))     #method 1
print(x.sum())       #method 2


# <h2>Sum by rows and by columns: </h2>

# In[19]:


x = np.array([[1, 1], [2, 2]])
x


# In[21]:


from IPython.display import Image
Image("path_to_image\\reductions.png")


# In[22]:


x.sum(axis=0)   # columns (first dimension)


# In[23]:


x[:, 0].sum(), x[:, 1].sum()   


# In[25]:


x.sum(axis=1)   # rows (second dimension)


# In[26]:


x[0, :].sum(), x[1, :].sum()


# In[84]:


np.random.seed( 10 )
x = np.random.rand(2, 3, 4)
x


# <h2>Important concepts related to sum along axis in numpy</h2>

# ***
# Here shape of an array x is (2,3,4)
# With axis=0, it sums along the 1st dimension, effectively removing it, leaving us with a 3x4 array. 
# 0.77132064 + 0.00394827 = 0.77526891, 0.02075195 + 0.51219226 = 0.53294421 etc.
# 
# Axis 1, condenses the size 3 dimension, result is 2x4. 0.77132064 + 0.49850701 + 0.16911084=1.43893849, etc.
# 
# Axis 2, condense the size 4 dimenson, result is 2x3. And sum is performed as below:
# 
# np.sum((0.77132064, 0.02075195, 0.63364823, 0.74880388)) = 2.17452471
# 
# np.sum((0.49850701, 0.22479665, 0.19806286, 0.76053071)) = 1.68189723
# 
# np.sum((0.16911084, 0.08833981, 0.68535982, 0.95339335)) = 1.89620382 and so on for the next row.
# ***

# In[88]:


print(x.sum(axis=0),"\n")
print(x.sum(axis=0).shape)


# In[93]:


print(x.sum(axis=1),"\n")

print(x.sum(axis=1).shape)


# In[92]:


print(x.sum(axis=2),"\n")

print(x.sum(axis=2).shape)


# In[44]:


x.sum(axis=2).shape


# In[46]:


x.sum(axis=2)


# In[103]:


#Let's consider this example:

x.sum(axis=2)[0, 1]


# <h2>Explanation</h2>

# In[102]:


#Here x.sum(axis=2) will sum with "axis=2" supressed. so array will be left with (2,3) shape. 
x.sum(axis=2)


# In[105]:


x.sum(axis=2)[0, 1] #[0,1] selects 1st row, 2nd element from above array.


# <h2>Array shape manipulation</h2>

# ## Flattening

# In[107]:


a = np.array([[1, 2, 3], [4, 5, 6]])
a


# In[108]:


a.T             #Transpose the array.


# In[109]:


a.T.ravel()                        


# ## Reshaping

# In[113]:


#The inverse operation to flattening:

a.shape


# In[115]:


b = a.ravel()
b = b.reshape((2,3))
b


# In[119]:


# or
a = np.array([[1, 2, 3], [4, 5, 6],[7,8,9],[10,11,12]])
a.reshape((2, -1))    # unspecified (-1) value is inferred


# ## Adding a dimension

# Indexing with the np.newaxis object allows us to add an axis to an array.

# In[124]:


z = np.array([1, 2, 3])
print(z)
print(z.shape)
print(z.ndim)


# In[126]:


print(z[:, np.newaxis], "\n")
print(z[:, np.newaxis].shape)
print(z[:, np.newaxis].ndim)


# In[128]:


print(z[np.newaxis, :])
print(z[np.newaxis, :].shape)
print(z[np.newaxis, :].ndim)


# ## Resizing

# Size of an array can be changed with ndarray.resize:

# In[137]:


a = np.arange(4)
a


# In[141]:


a.resize((8,1),refcheck=False)
a


# In[161]:


#help(np.ndarray.flatten)


# In[162]:


#help(np.ravel)


# ## Sorting data

# Sorting along an axis:

# In[168]:


a = np.array([[4, 3, 5], [1, 2, 1]])
b = np.sort(a, axis=1)                #Sorts each row separately!
b


# In[170]:


#In-place sort:
a.sort(axis=1)
a


# ## vstack and hstack

# numpy.hstack(tup) ----> Stack arrays in sequence horizontally (column wise). 
# 
# numpy.vstack(tup) ----> Stack arrays in sequence vertically (row wise).

# In[182]:


a = np.ones((3,3))
a


# In[189]:


# Now we want to add rows to above array

b = np.array((2,2,2))
np.vstack( (a, b) )  # or  np.vstack( (a, np.array((2,2,2))) )


# In[192]:


## Now we want to add rows to 'a' array

np.hstack( (a, b.reshape(3,1)) )


# ## numpy.stack

# numpy.stack(arrays, axis=0, out=None)
# 
# Join a sequence of arrays along a new axis.
# The axis parameter specifies the index of the new axis in the dimensions of the result. For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.

# In[196]:


a = np.array([[1,2],[3,4]]) 
b = np.array([[5,6],[7,8]]) 
print("array a:",a, "\n")
print("array b:",b)


# In[203]:


print('Stack the two arrays along axis 0:')
print(np.stack((a,b),0) )
print('\n')

print('Stack the two arrays along axis 2:')
print(np.stack((a,b),2))

