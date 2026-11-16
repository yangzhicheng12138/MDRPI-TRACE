import test from 'node:test';
import assert from 'node:assert/strict';
import {translate, setInterfaceLanguage, interfaceLanguage, bilingualTitle} from '../modes/frontend/i18n.js';

test('localizes exact interface labels and retains surrounding whitespace', () => {
  assert.equal(translate('  普通问答\n','en'), '  Chat\n');
  assert.equal(translate('普通问答','zh'), '普通问答');
});
test('keeps clinical text, numerical findings and unknown messages intact', () => {
  for (const text of ['SpO₂ 91%, CPAP 8 cmH₂O', '患者 56 岁，鼻梁红斑，按压可褪色。', '请评估普通问答中的病例。', '未知标题 · 3'])
    assert.equal(translate(text,'en'),text);
});
test('formats dynamic UI messages without translating captured user data', () => {
  assert.equal(translate('学习档案 · 张三','en'),'Learning profile · 张三');
  assert.equal(translate('提示 2','en'),'Hint 2');
  assert.equal(translate('符合条件：11 道练习','en'),'11 matching exercises');
  assert.equal(translate('成人 · 鼻导管','en'),'Adults · Nasal cannula');
});
test('language selection is explicit and bilingual titles respect it', () => {
  setInterfaceLanguage('en',{persist:false});
  assert.equal(bilingualTitle('练习 / Practice'),'Practice');
  assert.equal(setInterfaceLanguage('unsupported',{persist:false}),false);
  assert.equal(interfaceLanguage(),'en');
  setInterfaceLanguage('zh',{persist:false});
  assert.equal(bilingualTitle('练习 / Practice'),'练习');
  assert.equal(bilingualTitle('Original title'),'Original title');
});
