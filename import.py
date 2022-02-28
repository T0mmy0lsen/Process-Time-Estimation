import gc

from sql import SQL


def save(name, collection):
    i = 0
    for el in collection:
        i = i + len(el)
    i = i / len(collection)
    sql = SQL()
    for el in collection:
        if i == len(el):
            sql.insert(name, el)
        else:
            print("Length failure for {}".format(name))
            print(el)
    sql.commit()
    sql.close()


def convert():

    path = 'classification/input'
    batch_size = 100000

    for name in ['object', 'object_history', 'item', 'item_history', 'relation', 'relation_history']:

        filename = path + '/raw/' + name + '.txt'
        collection = []
        count = 1
        i = 0

        with open(filename, 'r', encoding='utf-16') as f:
            for line in f:
                if len(collection) == 0:
                    gc.disable()
                if i != 0:
                    collection.append(line.rstrip().split(';'))
                if len(collection) == batch_size:
                    save(name, collection)
                    del collection[:]
                    count = count + 1
                    gc.enable()
                i = i + 1
            save(name, collection)


def convert_special():

    path = 'classification/input'
    batch_size = 100000

    for name in ['communication', 'communication_history', 'request', 'request_history']:

        filename = path + '/raw/' + name + '.txt'
        collection = []
        count = 1
        i = 0

        with open(filename, 'r', encoding='utf-16') as f:
            buff = []
            for line in f:
                if len(collection) == 0:
                    gc.disable()
                if i != 0:
                    if line.startswith('[Entry]'):
                        str_line = ''.join(buff)
                        if (len(str_line) > 0) and str_line[-1] == '\n':
                            str_line = str_line.rstrip()
                            str_line = str_line[7:]
                            str_line_arr = str_line.split('[Delimiter]')
                            collection.append(str_line_arr)
                            buff = []
                    buff.append(line)
                if len(collection) == batch_size:
                    save(name, collection)
                    del collection[:]
                    count = count + 1
                    gc.enable()
                i = i + 1
            save(name, collection)


def main():
    convert()
    convert_special()
    return 0


if __name__ == '__main__':
    main()
