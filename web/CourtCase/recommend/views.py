from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

from .Core import CaseStatutesRecommend
from .collections import AllInfoCol


def index(request):
    return render(request, 'recommend/index.html')


def list_format(case_ids):
    # res = []
    # aic = AllInfoCol()
    #
    # for cid in case_ids:
    #     cs = aic.getSummary(cid)
    #     res.append({
    #         'id' : cid,
    #         'title' : cs,
    #     })

    res = [
        {
            'id' : '5a35c06f0e2c815b200b0809',
            'title' : '高某某婚约财产纠纷一案',
            'court' : '安徽省濉溪县人民法院',
            'time' : '2014',
            'keywords' : '彩礼 媒人 返还 解除婚约 收取 关系 婚约 相处 未果 过程',
            'causeOfAction' : '婚约财产纠纷'},
        {
            'id': '5a33abf30e2c810b3c383eca',
            'title': '刘某甲与郭某甲婚约财产纠纷一案',
            'court': ' 河南省睢县人民法院',
            'time': '2014',
            'keywords': '现金 回来 结婚仪式 所送 返还 彩礼 举行',
            'causeOfAction': '婚约财产纠纷'},
        {
            'id': '5a34608d0e2c810b3cd89d0e',
            'title': '张某甲与杨某某、张某乙等婚约财产纠纷一案',
            'court': ' 山东省临沭县人民法院',
            'time': '2014',
            'keywords': '衣服 摩托车 三金 财物 风俗 农村 结婚仪式 现金',
            'causeOfAction': '婚约财产纠纷'},
        {
            'id': '5a340d410e2c810b3c8de405',
            'title': '陈某与张某丙、张某乙等婚约财产纠纷一案',
            'court': '浙江省绍兴市越城区人民法院',
            'time': '2015',
            'keywords': '订婚 彩礼 金器 终止 证书 因故 仪式 返还',
            'causeOfAction': '婚约财产纠纷'},
    ]

    return res


def list(request):
    result = {}
    page_limit = 8
    query = str(request.GET.get('key'))
    print(query)

    # csr = CaseStatutesRecommend(query)
    # case_list, statutes_list = csr.recommend()

    case_list, statutes_list = [1,2,3], ['《最高人民法院关于适用〈中华人民共和国婚姻法〉若干问题解释（二）》第十条', '《中华人民共和国婚姻法》第三条','《中华人民共和国民事诉讼法》第一百四十四条', '《中华人民共和国民事诉讼法》第二百五十三条']
    pre_cases = list_format(case_list)

    paginator = Paginator(pre_cases, page_limit)

    page = request.GET.get('page', 1)

    try:
        cases = paginator.page(page)
    except PageNotAnInteger:
        cases = paginator.page(1)
    except EmptyPage:
        cases = paginator.page(paginator.num_pages)

    result['query'] = query
    result['cases'] = cases
    result['cases_num'] = len(cases)
    result['isPaging'] = len(pre_cases) > 6
    result['key'] = query
    result['statutes'] = statutes_list

    return render(request, 'recommend/list.html', result)


def display(request, case_id):
    result = {}

    # aic = AllInfoCol()
    # case = aic.getAllInfo(case_id)

    case = dict()
    case['caseplaintiffAlleges'] = '原告诉称：原、被某某于2015年1月18日举行结婚仪式，但未办理登记手续，后被某某于次日2015年1月19日，双方因家庭琐事发生口角，被某某回娘家，至今未回，原告于当日去娘家接被某某，被某某以种种理由拒不回家。原告认为，被某某刚举行结婚仪式一天便回娘家不归，其行为纯属于骗取原告彩礼款，故原告诉讼来院，要求1、被某某返还彩礼款及其他款共计18万元；2、案件受理费由被某某承担。'
    case['defendantArgued'] = '被某某辩称：2015年1月18日结婚，第二天早上，给我的压腰钱，原告向我要，我不给，原告的父母就把我们的结婚相给摔了，还把我打了，之后我就住院了，后来我就回娘家了，原告没有给我过18万元的彩礼，我没有骗婚。'
    case['factFound'] = '经审理查明：原、被某某双方经人介绍后，于2015年1月18日举行结婚仪式，但未进行婚姻登记。后于结婚仪式第二天即2015年1月19日，原、被某某因家庭琐事发生口角，被某某返回娘家至今未归。原告在举行结婚仪式前分两次先后给付被某某2万元、14万元彩礼款，原告向被某某索要彩礼款，被某某表示拒绝。'
    case['analysisProcess'] = '本院认为，1、自举行结婚仪式时起原、被某某之间就建立了婚约关系，最终双方无法缔结婚姻关系。依据《最高人民法院关于适用﹤中华人民共和国婚姻法﹥若干问题解释（二）》的相关规定，如果双方未办理结婚登记手续的，一方当事人请求另一方返还按照习俗给付的彩礼的，人民法院应当予以支持；2、原告主张其给付被某某彩礼16万元、一条金项链价值1万元，小包钱2000元及结婚当天给被某某的戴花钱1万元，故要求被某某返还彩礼共计18万元。并提供证人杜桂兰（媒人杜某甲作证，但庭审中证人杜桂兰表示其杜某甲付彩礼的过程，仅知道原告向被某某给付彩礼共人民币16万元，故原告提供的证据不能支持其向被某某给付的彩礼为18万元的主张。3、彩礼一般是指男女一方基于婚约关系，按照当地风俗习惯给付另一方数额较大的财物。综上，结合证人证言应确定原告向被某某给付的彩礼为人民币16万元。考虑被某某与原告已共同生活，且因家庭琐事导致双方最终无法缔结婚姻关系，故酌情确定被某某应向原告返还彩礼13万元为宜。 依照《中华人民共和国婚姻法》第三条、《最高人民法院关于适用﹤中华人民共和国婚姻法﹥若干问题的解释（二）》第十条第一款第（一）项之规定，判决如下：'
    case['caseDecision'] = '被某某王某某于判决生效后立即返还原告高某某彩礼款人民币130，000．00元。 如果未按本判决指定的期间履行给付金钱义务，应当按照《中华人民共和国民事诉讼法》第二百五十三条之规定，加倍支付迟延履行期间的债务利息。 案件受理费3，900．00元，由被某某承担。 如不服本判决，可在判决书送达之日起十五日内，向本院提交上诉状并按对方当事人的人数提出副本，上诉于吉林省长春市中级人民法院。'

    content = ''
    if case['caseplaintiffAlleges']:
        content += '  ' + case['caseplaintiffAlleges'] + '\n\n'
    if case['defendantArgued']:
        content += '  ' + case['defendantArgued'] + '\n\n'
    if case['factFound']:
        content += '  ' + case['factFound'] + '\n\n'
    if case['analysisProcess']:
        content += '  ' + case['analysisProcess'] + '\n\n'
    if case['caseDecision']:
        content += '  ' + case['caseDecision'] + '\n'

    result['case'] = dict(
        title = '高某某婚约财产纠纷一案',
        causeOfAction = '婚约财产纠纷',
        file = '吉林省农安县人民法院 民事判决书 （2015）农民初字第341号',
        content = content,
    )

    return render(request, 'recommend/display.html', result)