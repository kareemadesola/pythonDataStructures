def is_unique_compare_length(string):
    return len(set(string)) == len(string)


if __name__ == '__main__':
    print(is_unique_compare_length("werwer"))
