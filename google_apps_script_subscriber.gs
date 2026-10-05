/**
 * AI 台股研究中心｜訂閱名單共享服務
 *
 * 用途：
 * 1. Streamlit 網站 POST 新訂閱者。
 * 2. 將訂閱者保存到自動建立的 Google Sheet。
 * 3. Windows EmailAgent GET 同一份名單，寄送晨報／盤後分析。
 * 4. 新訂閱登記會寄通知給管理者。
 *
 * 使用前：
 * 1. 將 OWNER_EMAIL 改成你的管理者 Email。
 * 2. 將 SYNC_TOKEN 改成你自己設定的長字串。
 * 3. Deploy -> New deployment -> Web app。
 * 4. Web app URL 放進 Streamlit secrets / 本機 .env：
 *
 * SUBSCRIBER_SYNC_URL=https://script.google.com/macros/s/....../exec
 * SUBSCRIBER_SYNC_TOKEN=你的SYNC_TOKEN
 *
 * Google 官方文件：Web App 使用 Apps Script deployment 提供公開 URL；
 * 版本化 deployment 適合正式使用。
 */

const OWNER_EMAIL = 'YOUR_OWNER_EMAIL@example.com';
const SYNC_TOKEN = 'CHANGE_THIS_TO_A_LONG_RANDOM_TOKEN';
const SHEET_NAME = 'Subscribers';
const SPREADSHEET_PROPERTY = 'SUBSCRIBER_SPREADSHEET_ID';

function jsonOutput_(payload) {
  return ContentService
    .createTextOutput(JSON.stringify(payload))
    .setMimeType(ContentService.MimeType.JSON);
}

function authorized_(e, payload) {
  const queryToken = e && e.parameter ? String(e.parameter.token || '') : '';
  const bodyToken = payload && payload.token ? String(payload.token) : '';
  const token = queryToken || bodyToken;
  return token && token === SYNC_TOKEN && SYNC_TOKEN !== 'CHANGE_THIS_TO_A_LONG_RANDOM_TOKEN';
}

function ensureSheet_() {
  const props = PropertiesService.getScriptProperties();
  let spreadsheetId = props.getProperty(SPREADSHEET_PROPERTY);

  if (spreadsheetId) {
    try {
      const ss = SpreadsheetApp.openById(spreadsheetId);
      return ss.getSheetByName(SHEET_NAME) || ss.getSheets()[0];
    } catch (err) {
      // ID 不存在或已失效，重新建立。
    }
  }

  const ss = SpreadsheetApp.create('AI 台股研究中心｜訂閱名單');
  const sheet = ss.getSheets()[0];
  sheet.setName(SHEET_NAME);
  sheet.appendRow([
    'name',
    'email',
    'message',
    'subscribed',
    'active',
    'created_at',
    'updated_at',
    'source'
  ]);

  props.setProperty(SPREADSHEET_PROPERTY, ss.getId());
  return sheet;
}

function readSubscribers_() {
  const sheet = ensureSheet_();
  const values = sheet.getDataRange().getValues();
  if (values.length <= 1) return [];

  const rows = [];
  for (let i = 1; i < values.length; i++) {
    const r = values[i];
    if (!r[1]) continue;
    rows.push({
      name: String(r[0] || ''),
      email: String(r[1] || '').trim().toLowerCase(),
      message: String(r[2] || ''),
      subscribed: Boolean(r[3]),
      active: Boolean(r[4]),
      created_at: String(r[5] || ''),
      updated_at: String(r[6] || ''),
      source: String(r[7] || '')
    });
  }

  return rows.filter(r => r.active && r.subscribed && r.email);
}

function findRowByEmail_(sheet, email) {
  const values = sheet.getDataRange().getValues();
  const target = String(email || '').trim().toLowerCase();
  for (let i = 1; i < values.length; i++) {
    if (String(values[i][1] || '').trim().toLowerCase() === target) {
      return i + 1;
    }
  }
  return -1;
}

function validEmail_(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email || '').trim());
}

function doGet(e) {
  if (!authorized_(e, null)) {
    return jsonOutput_({ ok: false, error: 'unauthorized' });
  }

  return jsonOutput_({
    ok: true,
    subscribers: readSubscribers_()
  });
}

function doPost(e) {
  let payload = {};

  try {
    payload = JSON.parse((e && e.postData && e.postData.contents) || '{}');
  } catch (err) {
    return jsonOutput_({ ok: false, error: 'invalid_json' });
  }

  if (!authorized_(e, payload)) {
    return jsonOutput_({ ok: false, error: 'unauthorized' });
  }

  const action = String(payload.action || 'upsert');

  if (action === 'list') {
    return jsonOutput_({
      ok: true,
      subscribers: readSubscribers_()
    });
  }

  if (action !== 'upsert') {
    return jsonOutput_({ ok: false, error: 'unknown_action' });
  }

  const name = String(payload.name || '').trim().slice(0, 120);
  const email = String(payload.email || '').trim().toLowerCase().slice(0, 200);
  const message = String(payload.message || '').trim().slice(0, 5000);
  const subscribed = Boolean(payload.subscribed);
  const now = new Date().toISOString();

  if (!name || !email || !message || !validEmail_(email)) {
    return jsonOutput_({
      ok: false,
      error: 'invalid_fields'
    });
  }

  const sheet = ensureSheet_();
  const row = findRowByEmail_(sheet, email);
  let createdAt = now;
  let changed = true;

  if (row > 0) {
    const oldCreated = sheet.getRange(row, 6).getValue();
    if (oldCreated) createdAt = oldCreated;
    const oldActive = sheet.getRange(row, 5).getValue();
    const oldSubscribed = sheet.getRange(row, 4).getValue();
    changed = String(oldActive) !== String(subscribed) || String(oldSubscribed) !== String(subscribed);
    sheet.getRange(row, 1, 1, 8).setValues([[
      name,
      email,
      message,
      subscribed,
      subscribed,
      createdAt,
      now,
      String(payload.source || 'streamlit_subscription')
    ]]);
  } else {
    sheet.appendRow([
      name,
      email,
      message,
      subscribed,
      subscribed,
      createdAt,
      now,
      String(payload.source || 'streamlit_subscription')
    ]);
  }

  let ownerNotified = false;
  let notifyError = '';

  // 只有新登記或訂閱狀態有變時寄一次管理者通知。
  if (Boolean(payload.notify_owner) && (row < 0 || changed)) {
    try {
      if (!OWNER_EMAIL || OWNER_EMAIL.indexOf('@') < 1 || OWNER_EMAIL === 'YOUR_OWNER_EMAIL@example.com') {
        throw new Error('OWNER_EMAIL 尚未設定');
      }

      const subject = 'AI 台股研究中心｜新訂閱／聯絡登記';
      const body =
        '網站收到新的訂閱／聯絡資料。\n\n' +
        '姓名：' + name + '\n' +
        '電子郵件：' + email + '\n' +
        '訂閱晨報／盤後分析：' + (subscribed ? '是' : '否') + '\n\n' +
        '訊息：\n' + message + '\n\n' +
        '時間：' + now;

      MailApp.sendEmail(OWNER_EMAIL, subject, body);
      ownerNotified = true;
    } catch (err) {
      notifyError = String(err);
    }
  }

  return jsonOutput_({
    ok: true,
    saved: true,
    owner_notified: ownerNotified,
    error: notifyError
  });
}
