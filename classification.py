#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt


# In[2]:


from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


# In[3]:


digits = datasets.load_digits()


# In[4]:


_, axes = plt.subplots (nrows = 1, ncols = 4, figsize=(10, 3))


# In[9]:


for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)


# In[10]:


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# In[11]:


clf = svm.SVC(gamma=0.001)


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)


# In[13]:


clf.fit(x_train, y_train)


# In[14]:


predicted = clf.predict(x_test)


# In[16]:


_, axes = plt.subplots(nrows =1 , ncols=4 , figsize=(10, 3))


# In[18]:


for ax, image, prediction in zip(axes, x_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8,8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


# In[19]:


print(
    f"Clasification report for classifier {clf} : \n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


# In[ ]:




