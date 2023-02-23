import datetime


def date_string_to_datetime(date_string, strf='%Y-%m-%dT%H:%M:%S.%fZ'):
    if '.' not in date_string:
        # date_string2 = date_string.split('.')[0] + date_string[-1:]
        if '+' not in date_string:
            strf = '%Y-%m-%dT%H:%M:%SZ'
        else:
            if '+1000' in date_string:
                strf = '%Y-%m-%dT%H:%M+1000'
            elif '+1100' in date_string:
                strf = '%Y-%m-%dT%H:%M+1100'
            elif '+01:00' in date_string:
                strf = '%Y-%m-%dT%H:%M:%S+01:00'
            elif '+00:00' in date_string:
                strf = '%Y-%m-%dT%H:%M:%S+00:00'
    tmp = datetime.datetime.strptime(date_string, strf)
    date = tmp.strftime('%Y-%m-%d %H:%M:%S')
    return date


def get_month_max_day(month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    if month in [4, 6, 9, 11]:
        return 30


# 2021-07-16 00:00:0
def next_day(date):
    tmp0 = date.split(' ')
    tmp = tmp0[0].split('-')
    month = int(tmp[1])
    day = int(tmp[2])
    day += 1
    max_day = get_month_max_day(month)
    if day > max_day:
        day = 1
        month += 1
    return tmp[0] + '-' + str(month) + '-' + str(day) + ' ' + tmp0[1]
