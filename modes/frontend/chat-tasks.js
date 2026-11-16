export const CHAT_TASKS = [
  {id:'risk',icon:'shield',title:'评估MDRPI风险',note:'梳理人群、器械与受压线索',prompt:'请评估以下脱敏情境的MDRPI风险，区分已知风险线索与待补充信息。\n背景（人群、器械、使用时长、皮肤或黏膜表现）：\n'},
  {id:'tissue',icon:'search',title:'分析组织变化',note:'结合组织表现分析可能原因',prompt:'请分析以下器械接触部位的皮肤或黏膜变化，说明判断依据、需要鉴别的原因及尚不确定的信息。\n观察到的变化与器械接触经过：\n'},
  {id:'care',icon:'book',title:'制定护理与预防方案',note:'明确措施、适用条件与复评重点',prompt:'请根据以下脱敏情境制定MDRPI护理与预防方案，说明适用条件、器械治疗功能保障和复评重点。\n患者背景、器械、局部表现与当前处理：\n'},
  {id:'evidence',icon:'globe',title:'查询循证护理依据',note:'核对来源、年份和适用范围',prompt:'请查询以下MDRPI问题的循证护理依据，注明文献来源、年份、适用人群和器械，区分有证据支持的结论与不确定性。\n需要核对的问题及适用情境：\n'},
];
