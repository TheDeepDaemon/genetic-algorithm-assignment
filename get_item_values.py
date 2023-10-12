from load_config import get_item, NUM_ITEMS

def get_total_weight(item_vector):
    '''
        Just get the total weight, not value.
        Used for displaying the best chromosome
        after the genetic algorithm has been run.
    '''
    
    # confirm that it is a valid array
    assert(len(item_vector) == NUM_ITEMS)
    
    total_weight = 0

    for i, has_item in enumerate(item_vector):

        # check if the value for this item is true
        if has_item:

            # get the corresponding value and weight for this item
            weight, _ = get_item(i)

            # add to total
            total_weight += weight
    
    return total_weight

def get_active_items(item_vector):
    '''
        Get the list of items associated with this item vector.
        Used for displaying the best chromosome
        after the genetic algorithm has been run.
    '''

    # confirm that it is a valid array
    assert(len(item_vector) == NUM_ITEMS)

    all_items = []

    for i, has_item in enumerate(item_vector):

        # check if the value for this item is true
        if has_item:

            # get the corresponding value and weight for this item
            item = get_item(i)
            all_items.append(item)
    
    return len(all_items)
