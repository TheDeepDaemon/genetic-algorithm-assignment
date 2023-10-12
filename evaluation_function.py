from load_config import get_item, MAX_WEIGHT, NUM_ITEMS

def evaluate_backpack(item_vector):
    '''
        Determine the value of the backpack.
        Adds up the values, only returns the summed
        value when the weight is less than the threshold
        of 15.
    '''

    # confirm that it is a valid array
    assert(len(item_vector) == NUM_ITEMS)
    
    total_value = 0
    total_weight = 0

    for i, has_item in enumerate(item_vector):

        # check if the value for this item is true
        if has_item:

            # get the corresponding value and weight for this item
            weight, value = get_item(i)

            # add to total
            total_value += value
            total_weight += weight
    
    # only return the summed value if
    # the weight is not over the threshold
    # of MAX_WEIGHT
    if total_weight <= MAX_WEIGHT:
        return total_value
    else:
        return -total_weight
