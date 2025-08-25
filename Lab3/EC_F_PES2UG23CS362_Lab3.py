import numpy as np
from collections import Counter

def get_entropy_of_dataset(data: np.ndarray) -> float:
    """
    Calculate the entropy of the entire dataset using the target variable (last column).
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        float: Entropy value calculated using the formula: 
               Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        entropy = get_entropy_of_dataset(data)
        # Should return entropy based on target column ['yes', 'no', 'yes']
    """
    # TODO: Implement entropy calculation
    # Hint: Use np.unique() to get unique classes and their counts
    # Hint: Handle the case when probability is 0 to avoid log2(0)
    y = np.asarray(data) #explicitly convery to numpy array
    #np.unique() #gets unique classes and thier counts
    if y.ndim > 1:
        y = y[:, -1]
    
    if len(y) == 0:
        return 0.0

        
    # Count occurrences of each class
    counts = Counter(y) 
    probabilities = [count / len(y) for count in counts.values()] #here we calc propability for entropy calculation
        
    # Calculate entropy
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        
    return float(entropy)

    

def get_avg_info_of_attribute(data: np.ndarray, attribute: int) -> float:
    #weighted avg of all
    """
    Calculate the average information (weighted entropy) of a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate average information for
    
    Returns:
        float: Average information calculated using the formula:
               Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) 
               where S_v is subset of data with attribute value v
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        avg_info = get_avg_info_of_attribute(data, 0)  # For attribute at index 0
        # Should return weighted average entropy for attribute splits
    """
    # TODO: Implement average information calculation
    # Hint: For each unique value in the attribute column:
    #   1. Create a subset of data with that value
    #   2. Calculate the entropy of that subset
    #   3. Weight it by the proportion of samples with that value
    #   4. Sum all weighted entropies
    
    # data = np.asarray(data)
   
    # if data.size == 0:
    #     return 0.0
    
    # n_cols = data.shape[1] if data.ndim >1 else 1 #find no. of cols only when dimension is greater than 1
    
    # if attribute<0 or attribute >=n_cols -1: #ensure that attribute index is valid and its not refering to the target class
    #     raise IndexError(f"Attribute index {attribute} out of range")
    
    # total = data.shape[0]
    # weighted_sum = 0.0 

    # unique_vals, counts = np.unique(data[:,attribute], return_counts = True)
    # #get all unique values of chosen attribute and how many samples hv each value  - like for outlook - sunny rainy, overcast etc

    # for val, countt in zip(unique_vals, counts): #for each unique val and its freq
    #     subset = data[data[:,attribute]==val] #extract the subset of rows where attribute has the current val
    #     subset_entropy = get_entropy_of_dataset(subset)
    #     weighted_sum += (countt/total)*subset_entropy
    # return float(weighted_sum)
    data = np.asarray(data)
   
    if data.size == 0:
        return 0.0
    
    n_cols = data.shape[1] if data.ndim > 1 else 1
    
    if attribute < 0 or attribute >= n_cols - 1:
        raise IndexError(f"Attribute index {attribute} out of range")
    
    total = data.shape[0]
    weighted_sum = 0.0 

    # ✅ FIXED: use return_counts
    unique_vals, counts = np.unique(data[:, attribute], return_counts=True)

    for val, countt in zip(unique_vals, counts):
        subset = data[data[:, attribute] == val]
        subset_entropy = get_entropy_of_dataset(subset)
        weighted_sum += (countt / total) * subset_entropy

    return float(weighted_sum)

def get_information_gain(data: np.ndarray, attribute: int) -> float:
    """
    Calculate the Information Gain for a specific attribute.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
        attribute (int): Index of the attribute column to calculate information gain for
    
    Returns:
        float: Information gain calculated using the formula:
               Information_Gain = Entropy(S) - Avg_Info(attribute)
               Rounded to 4 decimal places
    
    Example:
        data = np.array([[1, 0, 'yes'],
                        [1, 1, 'no'],
                        [0, 0, 'yes']])
        gain = get_information_gain(data, 0)  # For attribute at index 0
        # Should return the information gain for splitting on attribute 0
    """
    # TODO: Implement information gain calculation
    # Hint: Information Gain = Dataset Entropy - Average Information of Attribute
    # Hint: Use the functions you implemented above
    # Hint: Round the result to 4 decimal places
    data = np.asarray(data)

    dataset_entropy = get_entropy_of_dataset(data)
    weighted_avg_entropy = get_avg_info_of_attribute(data,attribute)
    info_gain = dataset_entropy -weighted_avg_entropy #formula for information gain
    return float(info_gain)


def get_selected_attribute(data: np.ndarray) -> tuple:
    """
    Select the best attribute based on highest information gain.
    
    Args:
        data (np.ndarray): Dataset where the last column is the target variable
    
    Returns:
        tuple: A tuple containing:
            - dict: Dictionary mapping attribute indices to their information gains
            - int: Index of the attribute with the highest information gain
    
    Example:
        data = np.array([[1, 0, 2, 'yes'],
                        [1, 1, 1, 'no'],
                        [0, 0, 2, 'yes']])
        result = get_selected_attribute(data)
        # Should return something like: ({0: 0.123, 1: 0.456, 2: 0.789}, 2)
        # where 2 is the index of the attribute with highest gain
    """
    # TODO: Implement attribute selection
    # Hint: Calculate information gain for all attributes (except target variable)
    # Hint: Store gains in a dictionary with attribute index as key
    # Hint: Find the attribute with maximum gain using max() with key parameter
    # Hint: Return tuple (gain_dictionary, selected_attribute_index)
    data = np.asarray(data) 
    if data.size == 0: 
        return {}, -1
    gains = {} #dictionary of gains
    n_columns = data.shape[1] if data.ndim >1 else 1
    no_attri = n_columns -1 #excluding the target column

    for attr in range(no_attri):
        gains[attr] = get_information_gain(data, attr)
    selected = max(gains,key= gains.get) if gains else -1
    return gains, selected


def majority_class(y):
    #Return the majority class label from a list/array y.
    counts = Counter(y)
    return counts.most_common(1)[0][0]


def build_tree(data, attributes=None):
    
    #Recursively build decision tree using ID3.
    
   
    data = np.asarray(data)
    y = data[:, -1]
    
    # Stopping condition 1: all same class
    if len(set(y)) == 1:
        return {"label": y[0]}
    
    # Stopping condition 2: no attributes left
    n_cols = data.shape[1]
    if attributes is None:
        attributes = list(range(n_cols - 1))
    if not attributes:
        return {"label": majority_class(y)}
    
    # Select best attribute
    gains, best_attr = get_selected_attribute(data)
    if best_attr == -1:
        return {"label": majority_class(y)}
    
    tree = {"attribute": best_attr, "children": {}}
    
    # Recurse on each split
    for val in np.unique(data[:, best_attr]):
        subset = data[data[:, best_attr] == val]
        if subset.shape[0] == 0:
            tree["children"][val] = {"label": majority_class(y)}
        else:
            remaining_attrs = [a for a in attributes if a != best_attr]
            tree["children"][val] = build_tree(subset, remaining_attrs)
    
    return tree


def predict(tree, sample):
    
    #Predict the class label for a single sample using the decision tree.
    
    while "label" not in tree:
        attr = tree["attribute"]
        val = sample[attr]
        if val in tree["children"]:
            tree = tree["children"][val]
        else:
            # unseen value → return majority class of this node’s children
            labels = []
            for child in tree["children"].values():
                if "label" in child:
                    labels.append(child["label"])
            return majority_class(labels) if labels else None
    return tree["label"]


def predict_all(tree, data):
    
    #Predict labels for all rows in dataset using the decision tree.
    
    return [predict(tree, row) for row in data]
