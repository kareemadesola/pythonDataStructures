def get_third_and_fifth_element(list_of_tuple):
    return [list_of_tuple[2], list_of_tuple[4]]


def count_list_element_until_element_is_tuple(data_list):
    count = 0
    for element in data_list:
        if not isinstance(element, tuple):
            count += 1
        else:
            return count


# input must be a list of tuples whose element are pairs
# eg a =[(3,4),(5,6)]
def convert_python_tuple_to_dictionary(data_list):
    return dict(data_list)


def is_prime(num):
    if num > 1:
        for n in range(2, num):
            if (num % n) == 0:
                return False
        return True
    else:
        return False


def print_prime_numbers_between_1000_and_3000():
    return [i for i in range(1000, 3001) if is_prime(i)]
