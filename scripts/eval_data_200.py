"""Evaluation fixture: 200 beliefs and 50 queries.

This file is data only. `eval_recall.py` builds MemoryV2 rows from it; tests
check the contract (counts, unique labels, expected labels exist).

The fixture intentionally contains distractor clusters: several projects with
similar words, several time expressions, and multiple beliefs that can satisfy
one query. Expected sets are the labels a human would accept, not a single
canonical answer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Set, Tuple
from zoneinfo import ZoneInfo

TZ = "Asia/Shanghai"
Z = ZoneInfo(TZ)

CATEGORY_COUNTS = {
    "identity": 10,
    "explicit_facts": 15,
    "constraints": 20,
    "likes": 12,
    "dislikes": 10,
    "expertise": 12,
    "projects": 12,
    "goals": 15,
    "traits": 12,
    "interests": 12,
    "style_notes": 10,
    "behavior_patterns": 10,
    "decision_style": 10,
    "interpersonal": 10,
    "boundaries": 10,
    "habit": 10,
    "state": 10,
}

BELIEFS: List[Tuple[str, str, "float | None", str, str]] = []


def _add(category: str, labels: List[str]) -> None:
    for label in labels:
        due, kind, raw = DUE.get(label, (None, "none", ""))
        BELIEFS.append((category, label, due, kind, raw))


def _epoch(day: int, hour: int = 23, minute: int = 59) -> float:
    return datetime(2026, 10, day, hour, minute, tzinfo=Z).timestamp()


DUE: Dict[str, Tuple["float | None", str, str]] = {
    "周五要演示 Agent 平台": (_epoch(23), "due", "周五"),
    "下周三交周报": (_epoch(28), "due", "下周三"),
    "年底前换工作": (datetime(2026, 12, 31, 23, 59, tzinfo=Z).timestamp(), "range", "年底前"),
}

_add("identity", [
    "名字叫小陌", "英文名叫 Mo", "性别是男", "出生在浙江", "今年二十九岁",
    "在杭州住了五年", "养了一只叫团子的猫", "家里有个妹妹", "母语是中文", "右手写字",
])
_add("explicit_facts", [
    "住在杭州滨江", "在 SaaS 公司上班", "做后端开发", "每天坐地铁上班", "公司用飞书",
    "用 MacBook Pro", "手机是 iPhone", "大学学计算机", "毕业七年了", "有一只橘猫",
    "不喝咖啡", "会开车", "有驾照", "常用 Python", "周末通常在家",
])
_add("constraints", [
    "对花生过敏", "周三晚上不能开会", "周五下午要接孩子", "不能吃辣", "不能喝酒",
    "对猫毛轻微过敏", "晚上十点后不接电话", "周日不工作", "预算上限五千", "只能用中文沟通",
    "不接受远程办公", "不能出差超过三天", "对花粉过敏", "不吃海鲜", "不坐红眼航班",
    "不能熬夜", "会议不超过一小时", "不参加团建", "不用微信谈工作", "不接受临时改需求",
])
_add("likes", [
    "喜欢深色主题", "喜欢喝乌龙茶", "喜欢看悬疑片", "喜欢跑步", "喜欢下雨天", "喜欢猫",
    "喜欢简洁的 UI", "喜欢听播客", "喜欢做手冲咖啡", "喜欢独处", "喜欢纸质书", "喜欢杭州的秋天",
])
_add("dislikes", [
    "讨厌被叫全名", "讨厌嘈杂的餐厅", "讨厌开会", "讨厌临时改需求", "讨厌香菜",
    "讨厌电梯里的香水味", "讨厌无效社交", "讨厌别人迟到", "讨厌长语音", "讨厌被追问隐私",
])
_add("expertise", [
    "写后端", "熟悉 Python", "做过推荐系统", "会写 SQL", "懂 Kubernetes", "做过 RAG 项目",
    "会做数据建模", "懂支付系统", "会写技术文档", "做过性能优化", "熟悉 React", "会做代码评审",
])
_add("projects", [
    "在赶 Agent 平台项目", "在做 RAG 评测工具", "在迁移旧订单系统", "在做权限重构",
    "在搭内部知识库", "在写支付对账服务", "在做搜索排序优化", "在接企业微信",
    "在做数据看板", "在写移动端接口", "在做灰度发布系统", "在整理 API 文档",
])
_add("goals", [
    "周五要演示 Agent 平台", "下周三交周报", "年底前换工作", "这个月瘦四斤", "下个月考驾照",
    "明年读一个硕士", "月底前写完迁移脚本", "本周做完权限重构", "下季度带团队", "今年存十万",
    "学会做手冲咖啡", "每周跑三次步", "月底前还清信用卡", "下个月去云南旅行", "年底前搬新家",
])
_add("traits", [
    "做事偏谨慎", "不太爱表达情绪", "对细节敏感", "容易焦虑", "比较慢热", "有强迫症倾向",
    "喜欢提前计划", "不喜欢冒险", "对朋友很仗义", "有点完美主义", "说话直接", "容易心软",
])
_add("interests", [
    "对摄影感兴趣", "喜欢研究咖啡", "关注 AI 产品", "喜欢看纪录片", "对心理学感兴趣",
    "喜欢徒步", "关注开源社区", "喜欢逛博物馆", "对城市规划感兴趣", "喜欢看足球",
    "关注独立游戏", "喜欢做木工",
])
_add("style_notes", [
    "说话很短", "回复偏正式", "喜欢用表情包", "不用感叹号", "喜欢先给结论",
    "说话带一点自嘲", "不喜欢寒暄", "喜欢用英文术语", "打字有错别字", "喜欢列清单",
])
_add("behavior_patterns", [
    "经常熬夜到两点", "早上先看邮件", "开会前会列提纲", "周末常去咖啡馆", "下班后去跑步",
    "睡前刷手机", "遇到问题先查文档", "喜欢把任务拆小", "经常同时开很多标签页", "每周日整理待办",
])
_add("decision_style", [
    "决策前会列优缺点", "更相信数据", "不喜欢拍脑袋", "会先做小范围试验", "买大件前会研究很久",
    "倾向选熟悉的方案", "会听朋友意见", "决策后不常反悔", "对风险很敏感", "喜欢快速试错",
])
_add("interpersonal", [
    "和同事关系不错", "有一个认识十年的朋友", "和父母联系不多", "带过三个实习生",
    "不喜欢和领导走太近", "女朋友叫小林", "和邻居不熟", "经常帮朋友修电脑",
    "在团队里比较安静", "对陌生人客气",
])
_add("boundaries", [
    "不接受全远程岗位", "不聊政治", "不讨论工资", "不接受下班后开会", "不借钱给朋友",
    "不参加酒局", "不问别人年龄", "不分享定位", "不接受临时加需求", "不把工作手机带进卧室",
])
_add("habit", [
    "每天早上喝一杯乌龙茶", "晚上十一点后不看工作消息", "周一早上开周会", "周五下午整理本周记录",
    "睡前读十分钟书", "周末早上跑步", "每天午饭后散步", "每月底复盘", "通勤时听播客",
    "晚上十点后不喝咖啡",
])
_add("state", [
    "这周比较累", "最近在赶项目", "今天心情不错", "昨晚没睡好", "最近有点焦虑",
    "这周在出差", "今天在家办公", "最近在学 Kubernetes", "这周日程很满", "最近在控制饮食",
])

QUERIES: List[Tuple[str, Set[str]]] = [
    ("我那个 deadline 是几号", {"周五要演示 Agent 平台", "下周三交周报"}),
    ("接下来有什么安排", {"周五要演示 Agent 平台", "下周三交周报"}),
    ("周五要做什么", {"周五要演示 Agent 平台"}),
    ("周报什么时候交", {"下周三交周报"}),
    ("年底前有什么计划", {"年底前换工作"}),
    ("我最近有什么目标", {"这个月瘦四斤", "下个月考驾照"}),
    ("月底之前要完成什么", {"月底前写完迁移脚本", "月底前还清信用卡"}),
    ("下个月有什么安排", {"下个月考驾照", "下个月去云南旅行"}),
    ("这周要完成什么", {"本周做完权限重构"}),
    ("什么时候去旅行", {"下个月去云南旅行"}),
    ("我叫什么", {"名字叫小陌"}),
    ("我住哪儿", {"住在杭州滨江", "在杭州住了五年"}),
    ("我做什么工作", {"在 SaaS 公司上班", "做后端开发"}),
    ("我怎么上班", {"每天坐地铁上班"}),
    ("我养了什么宠物", {"养了一只叫团子的猫", "有一只橘猫"}),
    ("我大学学什么", {"大学学计算机"}),
    ("我平时用什么手机", {"手机是 iPhone"}),
    ("我用什么电脑", {"用 MacBook Pro"}),
    ("我会开车吗", {"会开车", "有驾照"}),
    ("我家里有什么人", {"家里有个妹妹"}),
    ("我对什么过敏", {"对花生过敏", "对花粉过敏"}),
    ("我不能吃什么", {"不能吃辣", "不吃海鲜"}),
    ("我喝酒吗", {"不能喝酒"}),
    ("我晚上有什么限制", {"晚上十点后不接电话", "不能熬夜"}),
    ("我喜欢喝什么", {"喜欢喝乌龙茶", "喜欢做手冲咖啡"}),
    ("我喜欢什么电影", {"喜欢看悬疑片"}),
    ("我讨厌什么", {"讨厌被叫全名", "讨厌香菜"}),
    ("我对工作有什么要求", {"不接受远程办公", "不接受临时改需求"}),
    ("我的预算是多少", {"预算上限五千"}),
    ("我周末通常做什么", {"周末通常在家", "周末常去咖啡馆"}),
    ("Agent 平台项目进展", {"在赶 Agent 平台项目", "周五要演示 Agent 平台"}),
    ("RAG 评测工具", {"在做 RAG 评测工具", "做过 RAG 项目"}),
    ("订单系统迁移", {"在迁移旧订单系统", "月底前写完迁移脚本"}),
    ("权限重构", {"在做权限重构", "本周做完权限重构"}),
    ("支付系统", {"在写支付对账服务", "懂支付系统"}),
    ("搜索排序", {"在做搜索排序优化"}),
    ("知识库", {"在搭内部知识库"}),
    ("数据看板", {"在做数据看板"}),
    ("灰度发布", {"在做灰度发布系统"}),
    ("API 文档", {"在整理 API 文档", "会写技术文档"}),
    ("我是什么性格", {"做事偏谨慎", "比较慢热"}),
    ("我怎么做决定", {"决策前会列优缺点", "更相信数据"}),
    ("我的朋友关系", {"有一个认识十年的朋友", "和同事关系不错"}),
    ("我有什么边界", {"不聊政治", "不讨论工资"}),
    ("我说话什么风格", {"说话很短", "喜欢先给结论"}),
    ("我平时几点睡", {"经常熬夜到两点", "昨晚没睡好"}),
    ("我最近状态", {"这周比较累", "最近有点焦虑"}),
    ("我有什么兴趣爱好", {"对摄影感兴趣", "喜欢徒步"}),
    ("我最近在学什么", {"最近在学 Kubernetes"}),
    ("我每天早上的习惯", {"每天早上喝一杯乌龙茶", "早上先看邮件"}),
]

LABELS = [label for _category, label, _due, _kind, _raw in BELIEFS]
LABEL_SET = set(LABELS)
CATEGORY_BY_LABEL = {label: category for category, label, _due, _kind, _raw in BELIEFS}


def validate() -> None:
    if len(BELIEFS) != 200:
        raise AssertionError(f"expected 200 beliefs, got {len(BELIEFS)}")
    if len(QUERIES) != 50:
        raise AssertionError(f"expected 50 queries, got {len(QUERIES)}")
    if len(LABEL_SET) != len(LABELS):
        duplicates = sorted({label for label in LABELS if LABELS.count(label) > 1})
        raise AssertionError(f"duplicate labels: {duplicates}")
    for category, expected in CATEGORY_COUNTS.items():
        actual = sum(1 for row in BELIEFS if row[0] == category)
        if actual != expected:
            raise AssertionError(f"{category}: expected {expected}, got {actual}")
    for query, expected_labels in QUERIES:
        missing = expected_labels - LABEL_SET
        if missing:
            raise AssertionError(f"{query!r}: missing labels {sorted(missing)}")


validate()
