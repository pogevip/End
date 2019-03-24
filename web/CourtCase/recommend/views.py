from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.conf import settings

from time import clock

from .collections import paragraph, lawcase
from .perform import searchStatuteEvaluateDisplay
from .CaseReasonPredict import roughExtract


def index(request):
    return render(request, 'recommend/index.html')


def list_format(cases, option):
    res = []
    par = paragraph()

    for case in cases:
        c = par.getInfoByFullTextId(case)
        res.append(dict(
            id=str(c[0]),
            title=c[1],
        ))

    return res


def list(request):
    result = {}
    page_limit = 10
    query = str(request.GET.get('key'))
    option = str(request.GET.get('option'))
    print(option)

    print("enter rough")
    startSeg = clock()
    roughRes = None

    if option == '关键字':
        roughRes = roughExtract(query).getIndexListbykeyword()
    elif option == 'TFIDF':
        roughRes = roughExtract(query).getIndexListbytfidf()
    elif option == 'LDA':
        roughRes = roughExtract(query).getIndexListbyLda()
    else:
        roughRes = roughExtract(query).getIndexListbykeyword()

    finishSeg = clock()
    print("索引耗时： %d 微秒" % (finishSeg - startSeg))
    # print(roughRes)
    # print("enter point")
    # pointRes = point(roughRes).getRes()
    # print(pointRes)
    pointRes = roughRes

    startPage = clock()
    pre_cases = list_format(pointRes, option)

    paginator = Paginator(pre_cases, page_limit)

    page = request.GET.get('page', 1)

    try:
        cases = paginator.page(page)
    except PageNotAnInteger:
        cases = paginator.page(1)
    except EmptyPage:
        cases = paginator.page(paginator.num_pages)

    result['query'] = query
    result['tactics_option'] = TACTICS_OPTIONS
    result['option'] = option
    result['cases'] = cases
    result['cases_num'] = len(cases)
    result['isPaging'] = len(pre_cases) > 6
    result['key'] = query
    finishPage = clock()
    print("分页耗时： %d 微秒" % (finishPage - startPage))

    return render(request, 'recommend/list.html', result)


def display(request, case_id):
    print(case_id)
    result = {}
    par = paragraph()
    lc = lawcase()

    case = par.getInfoByFullTextId(case_id)
    text = lc.getInfo(case_id)

    result['case'] = dict(
        title=case[1],
        content=text,
    )

    return render(request, 'recommend/display.html', result)


def getcoverCount(standard, test):
    count = 0
    for item in test:
        if item in standard:
            count += 1
    return count


def gettestresult(searchRes, option, col, referenceStandard):
    idlist = None
    if option != "test":
        idlist = [r['id'] for r in list_format(searchRes, option)]
    else:
        idlist = searchRes

    result = []
    for id in idlist:
        case = dict()
        case['id'] = id
        reference = [ref['name'].strip() + ref['levelone'].strip() \
                     for ref in col.find_one({"fullTextId": id})['references']]
        case['ref'] = reference
        case['covercount'] = getcoverCount(referenceStandard, reference)
        case['sim1'] = round(case['covercount']/(len(referenceStandard)+len(case['ref'])-case['covercount']), 4)
        P = case['covercount']/len(case['ref'])
        R = case['covercount']/len(referenceStandard)
        case['sim2'] = round(2 * P * R / (P + R), 4) if (P+R) != 0 else 0
        result.append(case)
    return result