const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const http = require('http');

const PORT = 8000;
let mainWindow = null;
let serverProcess = null;

// 패키징 여부에 따라 프로젝트 루트 경로 결정
const projectRoot = app.isPackaged
  ? path.join(process.resourcesPath, 'app')
  : path.join(__dirname, '..');

// ── FastAPI 서버 실행 ──────────────────────────────────────
function startServer() {

  serverProcess = spawn('python', [
    '-m', 'uvicorn',
    'app.server:app',
    '--host', '127.0.0.1',
    '--port', String(PORT),
    '--no-access-log',
  ], {
    cwd: projectRoot,
    windowsHide: true,
  });

  serverProcess.stdout.on('data', d => console.log('[py]', d.toString().trim()));
  serverProcess.stderr.on('data', d => {
    const msg = d.toString();
    console.log('[py]', msg.trim());
    if (msg.includes('Application startup complete')) {
      openWindow();
    }
  });

  serverProcess.on('error', err => console.error('[server error]', err));
  serverProcess.on('exit', code => console.log('[server exit]', code));
}

// ── 서버 준비될 때까지 폴링 후 창 열기 ────────────────────
function waitAndOpen(retries = 30) {
  http.get(`http://127.0.0.1:${PORT}/api/status`, res => {
    if (res.statusCode === 200) openWindow();
    else retry(retries);
  }).on('error', () => retry(retries));
}

function retry(retries) {
  if (retries <= 0) {
    openWindow(); // 타임아웃 시 그냥 열기
    return;
  }
  setTimeout(() => waitAndOpen(retries - 1), 500);
}

// ── Electron 창 생성 ──────────────────────────────────────
function openWindow() {
  if (mainWindow) return;

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'Bitcoin RL',
    backgroundColor: '#0d1117',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    autoHideMenuBar: true,
    show: false,
  });

  mainWindow.loadURL(`http://127.0.0.1:${PORT}`);

  mainWindow.once('ready-to-show', () => mainWindow.show());

  // 외부 링크는 브라우저에서 열기
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── 로딩 화면 (서버 뜨는 동안) ────────────────────────────
function openSplash() {
  const splash = new BrowserWindow({
    width: 340,
    height: 200,
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    backgroundColor: '#0d1117',
    webPreferences: { nodeIntegration: false },
  });

  splash.loadURL(`data:text/html,<!DOCTYPE html><html><body style="margin:0;display:flex;align-items:center;justify-content:center;height:100vh;background:%23161b22;font-family:system-ui;color:%23e6edf3;flex-direction:column;gap:12px;"><div style="font-size:32px;font-weight:800;color:%23f7931a;">&#8383;</div><div style="font-size:15px;font-weight:600;">Bitcoin RL</div><div style="font-size:12px;color:%238b949e;">서버 시작 중...</div></body></html>`);

  return splash;
}

// ── 앱 생명주기 ──────────────────────────────────────────
app.whenReady().then(() => {
  const splash = openSplash();
  startServer();
  waitAndOpen(40);

  // 창 열리면 스플래시 닫기
  const checkInterval = setInterval(() => {
    if (mainWindow) {
      clearInterval(checkInterval);
      splash.close();
    }
  }, 300);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (serverProcess) {
    serverProcess.kill();
    serverProcess = null;
  }
});
