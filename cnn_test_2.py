import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import jax.numpy as jnp
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import jax.numpy as jnp

digits = datasets.load_digits()

plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()

num_digits = len(digits.images)
print(f"Number of digits in the dataset: {num_digits}")

unique, counts = np.unique(digits.target, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class distribution:", class_distribution)

fig, axs = plt.subplots(1, 5, figsize=(10, 3))
classes_to_view = [0, 1, 2, 3, 4]
for i, cls in enumerate(classes_to_view):
    # Get the first image of each class
    idx = np.where(digits.target == cls)[0][0]
    axs[i].imshow(digits.images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
    axs[i].set_title(f'Class {cls}')
    axs[i].axis('off')

plt.show()

classes_to_keep = [0, 1, 3, 4]
indices_to_keep = np.isin(digits.target, classes_to_keep)

filtered_images = digits.images[indices_to_keep]
filtered_labels = digits.target[indices_to_keep]

# Print the number of images and class distribution in the filtered dataset
num_filtered_images = len(filtered_images)
print(f"Number of filtered images: {num_filtered_images}")

unique_filtered, counts_filtered = np.unique(filtered_labels, return_counts=True)
filtered_class_distribution = dict(zip(unique_filtered, counts_filtered))
print("Filtered class distribution:", filtered_class_distribution)
print(f"filtered images shape: {filtered_images.shape}")
print(f"filtered labels shape: {filtered_labels.shape}")

X = filtered_images.reshape(-1, 8, 8, 1)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(filtered_labels)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

def inverse_grid_number(n, number):
    if 1 <= number <= n**2:
        row_index = (number - 1) // n + 1
        column_index = (number - 1) % n + 1
        return row_index, column_index
    else:
        raise ValueError("Number must be between 1 and n^2 inclusive.")

def grid_number(n, a, b):
    if 1 <= a <= n and 1 <= b <= n:
        return (a - 1) * n + b
    else:
        raise ValueError("Row and column indices must be between 1 and n inclusive.")

def reflection_grid(n, coordinates):
    a, b = coordinates
    reflected_b = n - b + 1
    return a, reflected_b

def rotation_grid(n, coordinates):
    a, b = coordinates
    rotated_a = n - b + 1
    rotated_b = a
    return rotated_a, rotated_b

def rotate(n, number):
    (a,b) = inverse_grid_number(n, number)
    (new_a, new_b) = rotation_grid(n, (a,b))
    return grid_number(n, new_a, new_b)

def reflect(n, number):
    (a,b) = inverse_grid_number(n, number)
    (new_a, new_b) = reflection_grid(n, (a,b))
    return grid_number(n, new_a, new_b)

def generate_rotation_matrix(n):
    # Define the size of the matrix
    matrix_size = n ** 2

    # Initialize a matrix with zeros
    rotation_matrix = jnp.zeros((matrix_size, matrix_size), dtype=int)

    # Set 1 at the specified positions for each column
    for m in range(1, matrix_size + 1):
        rotated_position = rotate(n, m)
        rotation_matrix = rotation_matrix.at[rotated_position - 1, m - 1].set(1)  # Adjust for 0-based indexing

    return rotation_matrix

def generate_reflection_matrix(n):
    # Define the size of the matrix
    matrix_size = n ** 2

    # Initialize a matrix with zeros
    reflection_matrix = jnp.zeros((matrix_size, matrix_size), dtype=int)

    # Set 1 at the specified positions for each column
    for m in range(1, matrix_size + 1):
        reflected_position = reflect(n, m)
        reflection_matrix = reflection_matrix.at[reflected_position - 1, m - 1].set(1)  # Adjust for 0-based indexing

    return reflection_matrix

def generate_d4_matrices(n):
    '''Outputs n^2 by n^2 matrices'''
    # Get rotation and reflection matrices
    R = generate_rotation_matrix(n)
    S = generate_reflection_matrix(n)

    # Calculate R^2, R^3, SR, SR^2, SR^3
    R2 = jnp.dot(R, R)
    R3 = jnp.dot(R2, R)
    SR = jnp.dot(S, R)
    SR2 = jnp.dot(S, R2)
    SR3 = jnp.dot(S, R3)

    # Generate D4 matrices
    D4_matrices = [jnp.eye(n**2), R, R2, R3, S, SR, SR2, SR3]

    return D4_matrices



# In[9]:


def apply_transformation(image, transformation_matrix, n):
    flat_image = image.flatten()
    transformed_flat_image = jnp.dot(transformation_matrix, flat_image)
    return transformed_flat_image.reshape((n, n))

def apply_transformation_tf(image, matrix, n):
    flat_image = tf.reshape(image, [n**2])
    transformed_flat_image = tf.linalg.matvec(matrix, flat_image)
    transformed_image = tf.reshape(transformed_flat_image, [n, n, 1])
    return transformed_image


# ## Augmenting the dataset

# In[10]:


# Generate D4 rotation matrices for 8x8 images
n = 8
d4_matrices = generate_d4_matrices(n)

# Create augmented dataset
augmented_images = []
augmented_labels = []

print(len(filtered_images))

for img, lbl in zip(filtered_images, filtered_labels):
    for matrix in d4_matrices:
        transformed_image = apply_transformation(img, matrix, 8)
        augmented_images.append(transformed_image)
        augmented_labels.append(lbl)


augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

print(f"augmented images shape: {augmented_images.shape}")
print(f"aumented labels shape: {augmented_labels.shape}")

# Print the number of images and class distribution in the augmented dataset
num_augmented_images = len(augmented_images)
print(f"Number of augmented images: {num_augmented_images}")

unique_augmented, counts_augmented = np.unique(augmented_labels, return_counts=True)
augmented_class_distribution = dict(zip(unique_augmented, counts_augmented))
print("Augmented class distribution:", augmented_class_distribution)


# In[11]:


augmented_images_reshaped = augmented_images.reshape(-1, 8, 8, 1)
print(augmented_images_reshaped.shape)
print(augmented_labels.shape)
X_train_aug, X_test_aug, Y_train_aug, Y_test_aug = train_test_split(augmented_images_reshaped, augmented_labels, test_size=0.2, random_state=42, stratify=augmented_labels)


# In[12]:


def apply_transformation_batch(inputs, matrix):
    transformed = tf.map_fn(
        lambda x: apply_transformation_tf(x, matrix, 8), inputs, fn_output_signature=tf.float32
    )
    return transformed


# In[13]:


class CustomConvLayer(layers.Layer):
    def __init__(self, kernel_size):
        super(CustomConvLayer, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = self.add_weight(
            shape=(kernel_size, kernel_size, 1, 1),
            initializer="random_normal",
            trainable=True,
        )
        self.D4_matrices = generate_d4_matrices(8)

    def call(self, inputs):
        batch_size, height, width, channels = (
            tf.shape(inputs)[0],
            inputs.shape[1],
            inputs.shape[2],
            inputs.shape[3],
        )
        convolved_results = []

        for matrix in self.D4_matrices:
            matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
            transformed_inputs = apply_transformation_batch(inputs, matrix)
            convolved = tf.nn.conv2d(
                transformed_inputs, self.kernel, strides=[1, 1, 1, 1], padding="SAME"
            )
            convolved_results.append(convolved)

        convolved_average = tf.reduce_mean(tf.stack(convolved_results), axis=0)
        return convolved_average


class CustomPoolingLayer(layers.Layer):
    def __init__(self, pool_size):
        super(CustomPoolingLayer, self).__init__()
        self.pool_size = pool_size

    def call(self, inputs):
        res = tf.nn.avg_pool(
            inputs, ksize=[1, self.pool_size[0], self.pool_size[1], 1], strides=[1, self.pool_size[0], self.pool_size[1], 1], padding="VALID"
        )
        return res


# In[14]:


kernel_size = 7

model = models.Sequential([
    CustomConvLayer(kernel_size=kernel_size),
    CustomPoolingLayer(pool_size=(4, 4)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=50, batch_size=32)
print("Model training complete.")
test_loss, test_acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", test_acc)

# Fitting to augmented data
aug_model = models.Sequential([
    CustomConvLayer(kernel_size=kernel_size),
    CustomPoolingLayer(pool_size=(4, 4)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax'),
])
# print("FITTING TO AUGMENTED DATA-------------------")
# aug_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# aug_model.fit(X_train_aug, Y_train_aug, epochs=10, batch_size=32)
# print("Model training complete.")
# test_loss, test_acc = aug_model.evaluate(X_test_aug, Y_test_aug)
# print("Test accuracy:", test_acc)


# In[15]:


def create_circulant_matrix(kernel, image_size=8):
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    circ_matrix_size = image_size * image_size
    circ_matrix = np.zeros((circ_matrix_size, circ_matrix_size))

    for i in range(image_size):
        for j in range(image_size):
            row = np.zeros((image_size, image_size))
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    ii = i + ki - pad
                    jj = j + kj - pad
                    if 0 <= ii < image_size and 0 <= jj < image_size:
                        row[ii, jj] = kernel[ki, kj]
            circ_matrix[i * image_size + j, :] = row.flatten()

    return circ_matrix


# In[16]:


model.layers[0].kernel.shape


# In[17]:


kernel_matrix = tf.reshape(model.layers[0].kernel, [kernel_size, kernel_size])
kernel_circulant_matrix = create_circulant_matrix(kernel_matrix)

d4_matrices = generate_d4_matrices(8)
rotated_kernels = []
for d4_matrix in d4_matrices:
    rotated_kernel = kernel_circulant_matrix @ d4_matrix
    rotated_kernels.append(rotated_kernel)

averaged_circulant_kernel = tf.reduce_mean(tf.stack(rotated_kernels), axis=0)


# In[18]:


averaged_circulant_kernel.shape


# In[19]:


def visualize_combined_matrix(matrix, title='Combined Transformation Matrix'):
    plt.figure(figsize=(20, 16))
    sns.heatmap(matrix, annot=False, fmt=".2f", cmap='viridis')
    plt.title(title)
    plt.xlabel('Output Dimension')
    plt.ylabel('Input Dimension')
    plt.show()


# In[20]:


visualize_combined_matrix(kernel_matrix, title='Kernel before circulant')


# In[21]:


visualize_combined_matrix(kernel_circulant_matrix, title='after circulant before rotation')


# In[22]:


visualize_combined_matrix(averaged_circulant_kernel, title='Circulant Matrix')


# In[23]:


X_test[:10].shape


# In[26]:


D4_matrices = generate_d4_matrices(8)
dummy_test = tf.convert_to_tensor(X_test[:5], dtype=tf.float32)
# for i in range(10):
#     dummy_test = tf.convert_to_tensor(X_test[i : i + 1], dtype=tf.float32)
#     res = np.allclose(
#         tf.map_fn(
#             convolved_3,
#             dummy_test,
#             fn_output_signature=tf.float32,
#         ),
#         tf.map_fn(
#             convolved_3,
#             apply_transformation_batch(
#                 dummy_test, tf.convert_to_tensor(D4_matrices[-2], dtype=tf.float32)
#             ),
#             fn_output_signature=tf.float32,
#         ),
#     )
#     print(res)


# In[27]:


def convolved_3(x):
    operator = tf.reduce_mean(tf.stack([kernel_circulant_matrix @ matrix for matrix in D4_matrices], axis=0), axis=0) 
    return tf.reshape(operator @ tf.reshape(x,(64,1)), (8,8,1))


# In[28]:


convolved_results = []
for matrix in D4_matrices:
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    transformed_inputs = apply_transformation_batch(dummy_test, matrix)
    convolved = tf.nn.conv2d(
        transformed_inputs, model.layers[0].kernel, strides=[1, 1, 1, 1], padding="SAME"
    )
    convolved_results.append(convolved)

convolved_average = tf.reduce_mean(tf.stack(convolved_results), axis=0)


# In[29]:


def single_conv_layer(test_data):
    convolved_results = []
    for matrix in D4_matrices:
        matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
        transformed_inputs = apply_transformation_batch(test_data, matrix)
        convolved = tf.nn.conv2d(
            transformed_inputs, model.layers[0].kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        convolved_results.append(convolved)

    convolved_average = tf.reduce_mean(tf.stack(convolved_results), axis=0)
    return convolved_average


# In[30]:


test_1 = single_conv_layer(dummy_test)
test_2 = single_conv_layer(apply_transformation_batch(dummy_test, tf.convert_to_tensor(D4_matrices[-2], dtype=tf.float32)))
np.allclose(test_1, test_2)


# In[31]:


convolved_results_2 = []
for matrix in D4_matrices:
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    transformed_inputs = apply_transformation_batch(dummy_test, matrix)
    convolved_2 = tf.map_fn(
    lambda x: kernel_circulant_matrix @ tf.reshape(x, (64,1)),
    transformed_inputs,
    fn_output_signature=tf.float32,
)
    convolved_results_2.append(tf.reshape(convolved_2,(-1, 8, 8, 1)))

convolved_average_2 = tf.reduce_mean(tf.stack(convolved_results_2), axis=0)


# In[42]:


convolved_average_3 = tf.map_fn(
        convolved_3,
        dummy_test,
        fn_output_signature=tf.float32,
    )


# In[44]:


print(tf.equal(convolved_average,convolved_average_2))

# print("Difference:\n", difference)
# print("Max Difference:", np.max(np.abs(difference)))


# In[45]:


print(np.allclose(np.array(convolved_average_2),np.array(convolved_average_3)))


# In[34]:


np.allclose(tf.map_fn(
        convolved_3,
        dummy_test,
        fn_output_signature=tf.float32,
    ),
    tf.map_fn(
        convolved_3,
        apply_transformation_batch(dummy_test, tf.convert_to_tensor(D4_matrices[-2], dtype=tf.float32)),
        fn_output_signature=tf.float32
    )
)


# In[ ]:





# This is just some code to test circulant matrices and how tf handles conv2d

# In[35]:


kernel_width = 5
kernel = np.random.rand(kernel_width, kernel_width)
circ_matrix = create_circulant_matrix(kernel)
print("Circulant Matrix Shape:", circ_matrix.shape)

# Example image
image = np.random.rand(8, 8)

# Vectorize the image
image_vector = image.flatten()

# Perform the convolution using matrix multiplication
convolved_vector = circ_matrix @ image_vector

# Reshape the result back to 8x8
convolved_image = convolved_vector.reshape((8, 8))

print("Convolved Image using Block-Circulant Matrix:\n", convolved_image)

# TensorFlow convolution for comparison
transformed_inputs = tf.reshape(image, (1, 8, 8, 1))
kernel_tf = tf.reshape(kernel, (kernel_width, kernel_width, 1, 1))

# Perform the convolution using TensorFlow
convolved_tf = tf.nn.conv2d(transformed_inputs, kernel_tf, strides=[1, 1, 1, 1], padding="SAME")

# Extract the result
convolved_tf = tf.squeeze(convolved_tf).numpy()

print("TensorFlow Convolved Image:\n", convolved_tf)

# Compare results
difference = convolved_image - convolved_tf
print("Difference:\n", difference)
print("Max Difference:", np.max(np.abs(difference)))


# In[36]:


test_image = np.ones((8, 8))
test_kernel = 2 * np.ones((7,7))
test_image


# In[37]:


test_image = tf.reshape(test_image, (1, 8, 8, 1))
test_kernel = tf.reshape(test_kernel, (7, 7, 1, 1))


# In[38]:


res = tf.nn.conv2d(test_image, test_kernel, strides=[1, 1, 1, 1], padding="SAME")


# In[39]:


res = tf.reshape(res, (8, 8))


# In[40]:


res

