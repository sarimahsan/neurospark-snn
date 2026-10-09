@echo off
echo ========================================================
echo   NeuroSpark - Spiking Neural Network Simulator
echo ========================================================
echo Starting local web server on http://localhost:8080 ...
echo Press Ctrl+C to stop.
echo.
node -e "const http = require('http'); const fs = require('fs'); const path = require('path'); const server = http.createServer((req, res) => { let p = '.' + req.url.split('?')[0]; if (p === './') p = './index.html'; const ext = path.extname(p); const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' }; fs.readFile(p, (err, data) => { if(err) { res.writeHead(404); res.end('Not Found'); } else { res.writeHead(200, {'Content-Type': mime[ext] || 'text/plain', 'Cache-Control': 'no-cache, no-store, must-revalidate', 'Pragma': 'no-cache', 'Expires': '0'}); res.end(data); } }); }); server.listen(8080, () => { console.log('NeuroSpark is running at http://localhost:8080/'); require('child_process').exec('start http://localhost:8080'); });"
pause
