from datetime import datetime

import sql
from objs.Relation import Relation


class Request:

    _fillables = ['id', 'description', 'subject', 'solution', 'receivedDate', 'solutionDate', 'deadline', 'priority']
    _relations = []

    @property
    def fillables(self):
        return self._fillables

    @property
    def weight(self):
        return int(getattr(self, 'priority'))

    @property
    def deadline_timestamp(self):
        return getattr(self, 'deadline').timestamp()

    def __init__(self):
        self.id = 0

    def get_request(self, request_id):
        query = "SELECT {} FROM `request` WHERE `id` = %s".format(
            ', '.join(["`{}`".format(e) for e in self._fillables]))
        el = sql.SQL().one(query, [request_id])
        for idx, e in enumerate(self._fillables):
            setattr(self, e, el[idx])

        return self

    def get_relation(self):
        query = "SELECT `id`, `leftId`, `leftType`, `rightId`, `rightType` FROM `relation_history` WHERE `leftId` = %s"
        for el in sql.SQL().all(query, [self.id]):
            self._relations.append(Relation(el))
        return self


class Requests:

    _fillables = Request().fillables
    _r = []

    @property
    def r(self):
        return self._r

    @property
    def fillables(self):
        return self._fillables

    def get_between(self, t1, t2):
        for el in self.get_between_sql(t1, t2):
            request = Request()
            for idx, e in enumerate(self._fillables):
                setattr(request, e, el[idx])
            self._r.append(request)
        return self

    def get_between_sql(self, t1, t2):
        query = "SELECT {} FROM `request` WHERE receivedDate >= %s and receivedDate < %s".format(
            ', '.join(["`{}`".format(e) for e in self._fillables]))
        return sql.SQL().all(query, [t1, t2])
