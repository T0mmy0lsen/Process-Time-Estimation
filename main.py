from files import Files


def main():
    s = Files(name='CommunicationHistory')

    print(len(s.df().index))


if __name__ == '__main__':
    main()
