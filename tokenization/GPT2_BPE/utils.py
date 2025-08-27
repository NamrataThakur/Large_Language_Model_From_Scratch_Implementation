


#Find the pair of byte (int,int) that are frequently present and store the count of occurences:
def get_most_common(l_tokens,freq_dict=None, reversed=False):
    frequency = {} if freq_dict is None else freq_dict 
    for pair in zip(l_tokens, l_tokens[1:]):
        frequency[pair] = frequency.get(pair,0) + 1

    #Sort the dictionary in the descending/ascending order of the value:
    sorted_frequency_dict = {key : value for key, value in sorted(frequency.items(), 
                                                              key= lambda frequency : frequency[1], 
                                                              reverse= reversed )
                                                              }
    return sorted_frequency_dict


#Create a function that will merge the most frequently occuring pair with a new token:
def tokens_merge(token_list, pair, new_token):
    merged_tokens = []
    i = 0
    #for i in range(len(token_list)):
    while i < len(token_list):
        #print('1st i',i)
        if i < len(token_list) -1 and token_list[i] == pair[0] and token_list[i+1] == pair[1]:
            merged_tokens.append(new_token)
            i = i + 2
            #print('2nd i',i)
        else:
            #print(token_list[i])
            merged_tokens.append(token_list[i])
            i = i + 1
    return merged_tokens

if __name__ == '__main__':

    int_tokens = list(map(int, "Namrata Thakur".encode('utf-8')))
    sorted_frequency_dict = get_most_common(int_tokens,True)
    print(sorted_frequency_dict)

    print(tokens_merge([5, 6, 6, 7, 9, 1], (6, 7), 99))
    print(len(tokens_merge([5, 6, 6, 7, 9, 1], (6, 7), 99)))