import datetime

from objs.Request import Requests
from helpers import *


def main():



    """
    rs = Requests().get_between(
        str(datetime.datetime(2015, 12, 16)),
        str(datetime.datetime(2015, 12, 17))
    )

    chains = Chains([[e] for e in list(range(1, len(rs.r) + 1))])
    jobs = Jobs([Job(w=el.weight, d=el.deadline_timestamp, p=1) for el in rs.r])

    a = Algorithms(chains, jobs).lowest_cost_last()
    :return:
    """

    pass


if __name__ == '__main__':
    main()
