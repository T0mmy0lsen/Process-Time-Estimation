import math


class Job:

    def __init__(self, w, p):
        self.w = w      # Weight
        self.p = p      # Processing time
        self.id = 0     # Identifier
        self.C = 2      # Completion time
        self.d = 1      # Due date
        self.r = 0.05   # Rate

    def set_id(self, i):
        self.id = i

    def get_ratio(self):
        return self.w / self.p

    def get_weighted_completion_time(self):
        return self.w * self.p

    def get_weighted_tardiness(self):
        return self.w * self.get_tardiness()

    def get_weighted_number_of_tardy_jobs(self):
        return self.w * self.get_unit_penalty()

    def get_discounted_weighted_completion_time(self):
        return self.w * (1 - pow(math.e, -(self.r * self.C)))

    def get_is_less(self, job):
        return self.get_ratio() < job.ratio()

    def get_is_greater(self, job):
        return self.get_ratio() > job.ratio()

    def get_lateness(self):
        return self.C - self.d

    def get_tardiness(self):
        return max(self.get_lateness(), 0)

    def get_unit_penalty(self):
        if self.C > self.d:
            return 1
        return 0


class Jobs:

    def __init__(self, jobs):
        self.j = []
        for el in jobs:
            self.add(el)

    def add(self, job):
        if job.id == 0:
            job.set_id(len(self.j) + 1)
        self.j.append(job)
        return self

    # Theorem 3.1.6
    def get_discounted_total_weighted_completion_time(self):
        total = 0.0
        for t in self.j:
            total = total + (total + t.get_sum())
        return total

    # Lemma 3.1.2
    def total_weighted_completion_time(self):
        total = 0.0
        for t in self.j:
            total = total + (total + t.sum())
        return total

    # Lemma 3.1.2
    def p_factor(self):
        target_sum = 0
        target_jobs = []
        for i in range(1, len(self.j) + 1):
            tmp_jobs = self.j[0:i]
            tmp_sum = sum(job.w for job in tmp_jobs) / sum(job.p for job in tmp_jobs)
            if tmp_sum > target_sum:
                target_sum = tmp_sum
                target_jobs = tmp_jobs
        return target_sum, target_jobs

    @staticmethod
    # Lemma 3.1.2
    def total_weighted_completion_time_greater(jobs_l, jobs_r):
        return (
            sum(job.w for job in jobs_l) / sum(job.p for job in jobs_l) >
            sum(job.w for job in jobs_r) / sum(job.p for job in jobs_r)
        )


class Chains:

    def __init__(self, c):
        self.c = c

    def add(self, index, job_id):
        # Len = 2, thus index 0 and 1 is available.
        # We set index 2, then we initiate up to that index.
        if len(self.c) < index:
            for i in range(index + 1 - len(self.c)):
                self.c.append([])

        self.c[index].append(job_id)
        return self

    def empty(self):
        empty = True
        for i in self.c:
            if len(i) > 0:
                empty = False
        return empty

    def no_successors(self):
        return [e[-1] for e in self.c]


class Algorithms:

    def __init__(self, chains, jobs):
        self.chains = chains
        self.jobs = jobs
        self.order = []

    def finalize(self, lst, alg):
        self.order = [el for el in lst]
        print(alg)
        print([el.id for el in self.order])
        return self

    def get_jobs_from_identifiers(self, chain):
        tmp = []
        for idx, el in enumerate(self.jobs.j):
            if el.id in chain:
                tmp.append(el)
        return tmp

    # Theorem 3.1.1 Weighted Shortest Processing Time (WSPT)
    # The WSPT rule is optimal for 1 || sum(w_j * C_j)
    def weighted_shortest_processing_time(self):
        lst = sorted(self.jobs.j, key=lambda x: x.get_weighted_completion_time(), reverse=True)
        return self.finalize(lst, 'Weighted Shortest Processing Time')

    # Theorem 3.1.6 Weighted Discounted Shortest Processing Time (WDSPT)
    # The WDSPT rule is optimal for 1 || sum(w_j * (1 - e^(r * p_j)))
    def weighted_discounted_shortest_processing_time(self):
        lst = sorted(self.jobs.j, key=lambda x: x.get_discounted_weighted_completion_time(), reverse=True)
        return self.finalize(lst, 'Weighted Discounted Shortest Processing Time')

    # Algorithm 3.1.4 Total Weighted Completion Time and Chains
    def total_weighted_completion_time_and_chains(self):
        lst = []
        while not self.chains.empty():
            target_index = 0
            target_jobs = []
            target_sum = 0
            for idx, el in enumerate(self.chains.c):
                lst = self.get_jobs_from_identifiers(el)
                tmp_sum, tmp_jobs = Jobs(lst).p_factor()
                if tmp_sum > target_sum:
                    target_index = idx
                    target_jobs = tmp_jobs
                    target_sum = tmp_sum
            for el in target_jobs:
                lst.append(el)
                self.chains.c[target_index].remove(el.id)
        return self.finalize(lst, 'Total Weighted Completion Time and Chains')

    # Algorithm 3.2.1 Lowest Cost Last
    def lowest_cost_last(self):
        J = []
        Jc = [e.id for e in self.jobs]
        Js = self.chains.no_successors()

        return self.finalize(J, 'Lowest Cost Last')


def main():

    chains = Chains([[1, 2, 3, 4], [5, 6, 7]])
    jobs = Jobs([Job(6, 3), Job(18, 6), Job(12, 6), Job(8, 5), Job(8, 4), Job(17, 8), Job(18, 10)])

    a1 = Algorithms(chains, jobs).weighted_shortest_processing_time()
    a2 = Algorithms(chains, jobs).total_weighted_completion_time_and_chains()
    a3 = Algorithms(chains, jobs).weighted_discounted_shortest_processing_time()




if __name__ == '__main__':
    main()
