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