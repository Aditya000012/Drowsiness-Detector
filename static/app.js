/* DrowsGuard — Main Application */

// ===== GLOBALS =====
let supabaseClient = null, currentUser = null;
let sessionMode='timed', sessionDuration=1800, earThreshold=0.25, alertDelay=2;
let sessionActive=false, sessionStartTime=0, sessionTimerInterval=null;
let currentEAR=0, isDrowsy=false, drowsyStartTime=null, alertCount=0;
let alertTimestamps=[], earHistory=[];
let beepInterval=null, faceMeshInstance=null, cameraInstance=null;
let drowsyEpisodeStart=null, maxDrowsyEpisode=0, minEAR=1, sumEAR=0, earCount=0;
let dashTrendChart=null, earChart=null, histAlertsChart=null, histEarChart=null;
let sessionSaved=false;

const LEFT_EYE=[362,385,387,263,373,380], RIGHT_EYE=[33,160,158,133,153,144];
const FACE_OVAL=[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];

// ===== AUDIO =====
let audioCtx=null;
function getAudioCtx(){if(!audioCtx){const A=window.AudioContext||window.webkitAudioContext;audioCtx=new A();}return audioCtx;}
function playBeep(){try{const ctx=getAudioCtx();if(ctx.state==='suspended')ctx.resume();const o=ctx.createOscillator(),g=ctx.createGain();o.type='square';o.frequency.value=880;g.gain.setValueAtTime(.35,ctx.currentTime);g.gain.exponentialRampToValueAtTime(.01,ctx.currentTime+.4);o.connect(g);g.connect(ctx.destination);o.start();o.stop(ctx.currentTime+.4);}catch(e){}}
document.addEventListener('click',()=>{try{getAudioCtx().resume();}catch(e){}},{once:true});

// ===== UTILS =====
const $=id=>document.getElementById(id);
function fmtTime(s){const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=Math.floor(s%60);return`${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;}
function fmtDuration(s){if(!s||s<=0)return'0s';if(s<60)return Math.round(s)+'s';if(s<3600)return Math.round(s/60)+'m';return Math.floor(s/3600)+'h '+Math.round((s%3600)/60)+'m';}

// ===== ROUTER =====
function showPage(page,sub){
  document.querySelectorAll('.page').forEach(p=>p.classList.add('hidden'));
  const map={landing:'page-landing',auth:'page-auth',app:'page-app',session:'page-session'};
  const el=$(map[page]);if(el)el.classList.remove('hidden');
  if(page==='auth'&&sub)showAuthTab(sub);
  if(page==='app')showView('dashboard');
}

function showView(view){
  document.querySelectorAll('.view').forEach(v=>v.classList.add('hidden'));
  const el=$('view-'+view);if(el)el.classList.remove('hidden');
  document.querySelectorAll('.sidebar-btn').forEach(b=>b.classList.toggle('active',b.dataset.view===view));
  if(view==='dashboard')loadDashboard();
  if(view==='history')loadHistory();
}

function showAuthTab(tab){
  $('tab-login').classList.toggle('active',tab==='login');
  $('tab-register').classList.toggle('active',tab==='register');
  $('form-login').classList.toggle('hidden',tab!=='login');
  $('form-register').classList.toggle('hidden',tab!=='register');
  const e=$('auth-error');if(e)e.classList.add('hidden');
}

function showError(msg){const el=$('auth-error');if(el){el.textContent=msg;el.classList.remove('hidden');}}

// ===== AUTH =====
async function initSupabase(){
  try{
    const res=await fetch('/api/config');const cfg=await res.json();
    if(!cfg.supabaseUrl||!cfg.supabaseAnonKey){console.warn('Supabase keys missing');return;}
    supabaseClient=window.supabase.createClient(cfg.supabaseUrl,cfg.supabaseAnonKey);
    supabaseClient.auth.onAuthStateChange((ev,session)=>{
      if(session){currentUser=session.user;onLogin();}
      else{currentUser=null;showPage('landing');}
    });
    const{data}=await supabaseClient.auth.getSession();
    if(data.session){currentUser=data.session.user;onLogin();}
  }catch(e){console.warn('Supabase init error:',e);}
}

function onLogin(){
  const name=currentUser.user_metadata?.full_name||currentUser.email?.split('@')[0]||'User';
  const un=$('user-name');if(un)un.textContent=name;
  const dn=$('dash-name');if(dn)dn.textContent=name;
  showPage('app');
}

async function handleLogin(e){
  e.preventDefault();
  if(!supabaseClient)return showError('Supabase not configured');
  const email=$('login-email').value,pw=$('login-password').value;
  const btn=$('login-btn');btn.textContent='Logging in...';btn.disabled=true;
  const{error}=await supabaseClient.auth.signInWithPassword({email,password:pw});
  btn.textContent='Log In';btn.disabled=false;
  if(error)showError(error.message);
}

async function handleRegister(e){
  e.preventDefault();
  if(!supabaseClient)return showError('Supabase not configured');
  const name=$('reg-name').value,email=$('reg-email').value,pw=$('reg-password').value,confirm=$('reg-confirm').value;
  if(pw!==confirm)return showError('Passwords do not match');
  if(pw.length<6)return showError('Password must be at least 6 characters');
  const btn=$('reg-btn');btn.textContent='Creating...';btn.disabled=true;
  const{error}=await supabaseClient.auth.signUp({email,password:pw,options:{data:{full_name:name}}});
  btn.textContent='Create Account';btn.disabled=false;
  if(error)showError(error.message);
  else showError('✅ Check your email to confirm your account!');
}

async function handleLogout(){
  if(supabaseClient)await supabaseClient.auth.signOut();
  stopDetection();currentUser=null;showPage('landing');
}

// ===== DASHBOARD =====
async function loadDashboard(){
  if(!supabaseClient||!currentUser)return;
  try{
    const{data:sessions}=await supabaseClient.from('sessions').select('*').eq('user_id',currentUser.id).order('created_at',{ascending:false});
    if(!sessions||!sessions.length){$('stat-sessions').textContent='0';$('stat-alerts').textContent='0';$('stat-avg-dur').textContent='0m';$('stat-best').textContent='-';return;}
    const total=sessions.length;
    const totalAlerts=sessions.reduce((s,r)=>s+(r.alert_count||0),0);
    const avgDur=Math.round(sessions.reduce((s,r)=>s+(r.actual_duration||0),0)/total);
    const best=sessions.reduce((b,r)=>(r.alert_count||0)<(b.alert_count||0)?r:b,sessions[0]);
    $('stat-sessions').textContent=total;$('stat-alerts').textContent=totalAlerts;
    $('stat-avg-dur').textContent=fmtDuration(avgDur);$('stat-best').textContent=(best.alert_count||0)+' alerts';
    const recent=sessions.slice(0,5);
    let html='<table class="data-table"><thead><tr><th>Date</th><th>Label</th><th>Mode</th><th>Duration</th><th>Alerts</th></tr></thead><tbody>';
    recent.forEach(r=>{html+=`<tr><td>${new Date(r.created_at).toLocaleDateString()}</td><td>${r.label||'—'}</td><td>${r.mode||'—'}</td><td>${fmtDuration(r.actual_duration)}</td><td>${r.alert_count||0}</td></tr>`;});
    html+='</tbody></table>';$('recent-table-wrap').innerHTML=html;
    const last10=sessions.slice(0,10).reverse();
    if(dashTrendChart)dashTrendChart.destroy();
    const ctx=$('dash-trend-chart');if(!ctx)return;
    dashTrendChart=new Chart(ctx,{type:'bar',data:{labels:last10.map(r=>r.label||'—'),datasets:[{label:'Alerts',data:last10.map(r=>r.alert_count||0),backgroundColor:'rgba(0,212,255,0.5)',borderColor:'#00d4ff',borderWidth:1,borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#7a8a9a',font:{size:10}},grid:{display:false}},y:{ticks:{color:'#7a8a9a'},grid:{color:'rgba(26,42,58,0.3)'},beginAtZero:true}}}});
  }catch(e){console.error('Dashboard error:',e);}
}

// ===== SESSION SETUP =====
function setLabel(l){const el=$('setup-label');if(el)el.value=l;}
function selectMode(m){sessionMode=m;$('mode-timed').classList.toggle('selected',m==='timed');$('mode-continuous').classList.toggle('selected',m==='continuous');$('duration-picker').classList.toggle('hidden',m==='continuous');}
function setDuration(s){sessionDuration=s;$('dur-hours').value=Math.floor(s/3600);$('dur-mins').value=Math.floor((s%3600)/60);}
function setDelay(el,d){alertDelay=d;el.closest('.preset-btns').querySelectorAll('.tag').forEach(t=>t.classList.remove('selected'));el.classList.add('selected');}

function beginSession(){
  const label=($('setup-label').value||'Untitled').trim();
  earThreshold=parseFloat($('setup-thresh').value);
  if(sessionMode==='timed'){sessionDuration=parseInt($('dur-hours').value||0)*3600+parseInt($('dur-mins').value||0)*60;if(sessionDuration<60){alert('Minimum duration is 1 minute');return;}}
  $('session-label-display').textContent=label;
  $('live-thresh').value=earThreshold;$('live-thresh-val').textContent=earThreshold.toFixed(2);
  $('timer-mode-label').textContent=sessionMode==='timed'?'TIMED MODE':'CONTINUOUS MODE';
  alertCount=0;alertTimestamps=[];earHistory=[];isDrowsy=false;drowsyStartTime=null;
  drowsyEpisodeStart=null;maxDrowsyEpisode=0;minEAR=1;sumEAR=0;earCount=0;sessionSaved=false;currentEAR=0;
  $('alert-count').textContent='0';$('alert-log').innerHTML='<p class="empty-msg">No alerts yet.</p>';
  $('status-badge').className='status-badge awake';$('status-text').textContent='✓ AWAKE';
  $('drowsy-border').classList.remove('active');$('ear-value').textContent='0.000';$('ear-value').classList.remove('low');
  $('no-face').classList.add('hidden');$('cam-error').classList.add('hidden');
  showPage('session');sessionActive=true;sessionStartTime=Date.now();
  initEarChart();startTimer();startDetection();
}

// ===== TIMER =====
function startTimer(){
  clearInterval(sessionTimerInterval);
  sessionTimerInterval=setInterval(()=>{
    if(!sessionActive)return;
    const elapsed=(Date.now()-sessionStartTime)/1000;
    if(sessionMode==='timed'){
      const remaining=Math.max(0,sessionDuration-elapsed);
      $('session-timer').textContent=fmtTime(remaining);
      if(remaining<300)$('session-timer').style.color='#ff8800';
      if(remaining<60)$('session-timer').style.color='var(--red)';
      if(remaining<=0)endSession();
    }else{
      $('session-timer').textContent=fmtTime(elapsed);
    }
  },1000);
}

// ===== EAR CHART =====
function initEarChart(){
  if(earChart)earChart.destroy();
  const ctx=$('ear-chart');if(!ctx)return;
  earChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:[{data:[],borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.08)',borderWidth:1.5,pointRadius:0,fill:true,tension:.3},{data:[],borderColor:'rgba(255,59,59,0.5)',borderWidth:1,borderDash:[4,4],pointRadius:0,fill:false}]},options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},scales:{x:{display:false},y:{min:.05,max:.5,ticks:{color:'#4a5a6a',font:{size:9,family:'JetBrains Mono'},stepSize:.1},grid:{color:'rgba(26,42,58,0.3)'}}},plugins:{legend:{display:false},tooltip:{enabled:false}}}});
}

function updateEarChart(){
  if(!earChart)return;
  earHistory.push(currentEAR);if(earHistory.length>120)earHistory.shift();
  earChart.data.labels=earHistory.map((_,i)=>i);
  earChart.data.datasets[0].data=[...earHistory];
  earChart.data.datasets[1].data=earHistory.map(()=>earThreshold);
  earChart.data.datasets[0].borderColor=isDrowsy?'#ff3b3b':'#00d4ff';
  earChart.data.datasets[0].backgroundColor=isDrowsy?'rgba(255,59,59,.08)':'rgba(0,212,255,.08)';
  earChart.update('none');
}

function updateLiveThreshold(v){earThreshold=parseFloat(v);$('live-thresh-val').textContent=parseFloat(v).toFixed(2);}

// ===== DETECTION ENGINE =====
function dist(a,b){return Math.sqrt((a.x-b.x)**2+(a.y-b.y)**2);}
function calcEAR(lm,idx,w,h){const p=idx.map(i=>({x:lm[i].x*w,y:lm[i].y*h}));const v1=dist(p[1],p[5]),v2=dist(p[2],p[4]),hz=dist(p[0],p[3]);return hz===0?0:(v1+v2)/(2*hz);}

async function startDetection(){
  const video=$('webcam'),canvas=$('overlay-canvas');if(!video||!canvas)return;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:640,height:480}});
    video.srcObject=stream;await video.play();$('cam-error').classList.add('hidden');
  }catch(e){$('cam-error').classList.remove('hidden');return;}

  faceMeshInstance=new FaceMesh({locateFile:f=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${f}`});
  faceMeshInstance.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:.5,minTrackingConfidence:.5});
  faceMeshInstance.onResults(results=>{
    if(!sessionActive)return;
    const W=video.videoWidth||640,H=video.videoHeight||480;
    canvas.width=W;canvas.height=H;
    const c=canvas.getContext('2d');c.clearRect(0,0,W,H);
    if(results.multiFaceLandmarks&&results.multiFaceLandmarks.length){
      $('no-face').classList.add('hidden');
      const lm=results.multiFaceLandmarks[0];
      const lEAR=calcEAR(lm,LEFT_EYE,W,H),rEAR=calcEAR(lm,RIGHT_EYE,W,H);
      currentEAR=(lEAR+rEAR)/2;sumEAR+=currentEAR;earCount++;
      if(currentEAR>0&&currentEAR<minEAR)minEAR=currentEAR;
      const below=currentEAR<earThreshold;
      const color=below?'#ff3b3b':'#00ff88',glow=below?'rgba(255,59,59,.25)':'rgba(0,255,136,.2)';
      drawEye(c,lm,LEFT_EYE,W,H,color,glow);drawEye(c,lm,RIGHT_EYE,W,H,color,glow);
      [...LEFT_EYE,...RIGHT_EYE].forEach(i=>{c.beginPath();c.arc(lm[i].x*W,lm[i].y*H,2.5,0,Math.PI*2);c.fillStyle='#00d4ff';c.fill();});
      c.beginPath();FACE_OVAL.forEach((idx,i)=>{const x=lm[idx].x*W,y=lm[idx].y*H;i===0?c.moveTo(x,y):c.lineTo(x,y);});c.closePath();c.strokeStyle='rgba(0,212,255,.12)';c.lineWidth=1;c.stroke();
      if(below){
        if(!drowsyStartTime)drowsyStartTime=Date.now();
        else if((Date.now()-drowsyStartTime)/1000>=alertDelay&&!isDrowsy){
          isDrowsy=true;alertCount++;if(!drowsyEpisodeStart)drowsyEpisodeStart=Date.now();
          const ts=new Date().toLocaleTimeString('en-US',{hour12:false});alertTimestamps.push(ts);
          $('alert-count').textContent=alertCount;addAlertLog(ts);
          if(!beepInterval){playBeep();beepInterval=setInterval(playBeep,800);}
        }
      }else{
        if(isDrowsy&&drowsyEpisodeStart){const dur=(Date.now()-drowsyEpisodeStart)/1000;if(dur>maxDrowsyEpisode)maxDrowsyEpisode=dur;drowsyEpisodeStart=null;}
        drowsyStartTime=null;isDrowsy=false;if(beepInterval){clearInterval(beepInterval);beepInterval=null;}
      }
    }else{
      $('no-face').classList.remove('hidden');
      if(isDrowsy&&drowsyEpisodeStart){const dur=(Date.now()-drowsyEpisodeStart)/1000;if(dur>maxDrowsyEpisode)maxDrowsyEpisode=dur;drowsyEpisodeStart=null;}
      drowsyStartTime=null;isDrowsy=false;currentEAR=0;if(beepInterval){clearInterval(beepInterval);beepInterval=null;}
    }
    updateSessionUI();updateEarChart();
  });

  try{
    cameraInstance=new Camera(video,{onFrame:async()=>{if(sessionActive&&faceMeshInstance)await faceMeshInstance.send({image:video});},width:640,height:480});
    cameraInstance.start();
  }catch(e){
    const loop=async()=>{if(!sessionActive)return;if(faceMeshInstance)await faceMeshInstance.send({image:video});requestAnimationFrame(loop);};
    loop();
  }
}

function drawEye(c,lm,idx,W,H,color,glow){
  c.beginPath();idx.forEach((id,i)=>{const x=lm[id].x*W,y=lm[id].y*H;i===0?c.moveTo(x,y):c.lineTo(x,y);});
  c.closePath();c.fillStyle=glow;c.fill();c.strokeStyle=color;c.lineWidth=1.5;c.stroke();
}

function updateSessionUI(){
  const earEl=$('ear-value');earEl.textContent=currentEAR.toFixed(3);earEl.classList.toggle('low',currentEAR>0&&currentEAR<earThreshold);
  const pct=Math.max(0,Math.min(100,((currentEAR-.05)/.4)*100));$('ear-bar').style.width=pct+'%';
  const threshPct=Math.max(0,Math.min(100,((earThreshold-.05)/.4)*100));$('ear-thresh-mark').style.left=threshPct+'%';
  if(isDrowsy){$('status-badge').className='status-badge drowsy';$('status-text').textContent='⚠ DROWSY';$('drowsy-border').classList.add('active');}
  else{$('status-badge').className='status-badge awake';$('status-text').textContent='✓ AWAKE';$('drowsy-border').classList.remove('active');}
}

function addAlertLog(ts){
  const log=$('alert-log');const empty=log.querySelector('.empty-msg');if(empty)empty.remove();
  const d=document.createElement('div');d.className='log-entry';
  d.innerHTML=`<span class="log-icon">⚠</span><span class="log-time">[${ts}]</span><span>Drowsy episode #${alertCount}</span>`;
  log.prepend(d);
}

function stopDetection(){
  sessionActive=false;
  if(cameraInstance){try{cameraInstance.stop();}catch(e){}}cameraInstance=null;
  if(faceMeshInstance){try{faceMeshInstance.close();}catch(e){}}faceMeshInstance=null;
  const video=$('webcam');if(video&&video.srcObject){video.srcObject.getTracks().forEach(t=>t.stop());video.srcObject=null;}
  if(beepInterval){clearInterval(beepInterval);beepInterval=null;}
  const border=$('drowsy-border');if(border)border.classList.remove('active');
}

// ===== END SESSION =====
async function endSession(){
  if(!sessionActive)return;
  sessionActive=false;clearInterval(sessionTimerInterval);
  if(isDrowsy&&drowsyEpisodeStart){const dur=(Date.now()-drowsyEpisodeStart)/1000;if(dur>maxDrowsyEpisode)maxDrowsyEpisode=dur;}
  stopDetection();
  const actualDuration=Math.round((Date.now()-sessionStartTime)/1000);
  const avgEar=earCount>0?sumEAR/earCount:0;
  const label=$('session-label-display').textContent||'Session';
  $('sum-label').textContent=label;$('sum-mode').textContent=sessionMode==='timed'?'Timed':'Continuous';
  $('sum-duration').textContent=fmtTime(actualDuration);$('sum-alerts').textContent=alertCount;
  $('sum-longest').textContent=maxDrowsyEpisode.toFixed(1)+'s';$('sum-ear').textContent=avgEar.toFixed(3);
  const tl=$('sum-timeline');tl.innerHTML='';
  if(actualDuration>0){
    const sessionStartSec=Math.floor(sessionStartTime/1000);
    alertTimestamps.forEach(ts=>{
      const parts=ts.split(':');const tSec=parseInt(parts[0])*3600+parseInt(parts[1])*60+parseInt(parts[2]);
      const dayStart=sessionStartSec-(sessionStartSec%86400);const offset=tSec+dayStart-sessionStartSec;
      const pct=Math.max(0,Math.min(98,(offset/actualDuration)*100));
      const dot=document.createElement('div');dot.className='timeline-dot';dot.style.left=pct+'%';dot.title=ts;tl.appendChild(dot);
    });
  }
  const sl=$('sum-log');sl.innerHTML='';
  if(alertTimestamps.length===0){sl.innerHTML='<p class="empty-msg">No drowsy episodes — great session! 🎉</p>';}
  else{alertTimestamps.forEach((ts,i)=>{const d=document.createElement('div');d.className='log-entry';d.innerHTML=`<span class="log-icon">⚠</span><span class="log-time">[${ts}]</span><span>Episode #${i+1}</span>`;sl.appendChild(d);});}
  if(supabaseClient&&currentUser&&!sessionSaved){
    sessionSaved=true;
    try{await supabaseClient.from('sessions').insert({user_id:currentUser.id,label,mode:sessionMode,planned_duration:sessionMode==='timed'?sessionDuration:null,actual_duration:actualDuration,alert_count:alertCount,avg_ear:parseFloat(avgEar.toFixed(4)),min_ear:parseFloat((minEAR===1?0:minEAR).toFixed(4)),max_drowsy_episode:parseFloat(maxDrowsyEpisode.toFixed(2)),alert_timestamps:alertTimestamps});}
    catch(e){console.error('Save error:',e);}
  }
  $('modal-summary').classList.remove('hidden');
}

function saveAndGoToDash(){$('modal-summary').classList.add('hidden');showPage('app');showView('dashboard');}
function saveAndGoToHistory(){$('modal-summary').classList.add('hidden');showPage('app');showView('history');}

// ===== HISTORY =====
async function loadHistory(){
  if(!supabaseClient||!currentUser)return;
  try{
    let query=supabaseClient.from('sessions').select('*').eq('user_id',currentUser.id).order('created_at',{ascending:false});
    const modeFilter=$('filter-mode').value;if(modeFilter)query=query.eq('mode',modeFilter);
    const labelFilter=$('filter-label').value;if(labelFilter)query=query.eq('label',labelFilter);
    const{data:sessions}=await query;if(!sessions)return;
    const allL=await supabaseClient.from('sessions').select('label').eq('user_id',currentUser.id);
    if(allL.data){
      const labels=[...new Set(allL.data.map(s=>s.label).filter(Boolean))];
      const sel=$('filter-label');const cur=sel.value;sel.innerHTML='<option value="">All Labels</option>';
      labels.forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=l;if(l===cur)o.selected=true;sel.appendChild(o);});
    }
    const tbody=$('history-tbody');
    if(!sessions.length){tbody.innerHTML='<tr><td colspan="6" class="empty-msg">No sessions found.</td></tr>';}
    else{
      tbody.innerHTML='';
      sessions.forEach(r=>{
        const tr=document.createElement('tr');
        tr.innerHTML=`<td>${new Date(r.created_at).toLocaleDateString()}</td><td>${r.label||'—'}</td><td style="text-transform:capitalize">${r.mode||'—'}</td><td>${fmtDuration(r.actual_duration)}</td><td style="color:${(r.alert_count||0)>5?'var(--red)':(r.alert_count||0)>0?'#ff8800':'var(--green)'}">${r.alert_count||0}</td><td style="font-family:var(--mono)">${(r.avg_ear||0).toFixed(3)}</td>`;
        tbody.appendChild(tr);
      });
    }
    const chrono=sessions.slice().reverse();
    const chartLabels=chrono.map(r=>r.label||'—');
    const baseOpts={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#7a8a9a',font:{size:10}},grid:{display:false}},y:{ticks:{color:'#7a8a9a'},grid:{color:'rgba(26,42,58,0.3)'},beginAtZero:true}}};
    if(histAlertsChart)histAlertsChart.destroy();
    const actx=$('hist-alerts-chart');if(actx)histAlertsChart=new Chart(actx,{type:'line',data:{labels:chartLabels,datasets:[{data:chrono.map(r=>r.alert_count||0),borderColor:'#ff3b3b',backgroundColor:'rgba(255,59,59,.1)',fill:true,tension:.3,pointRadius:3,pointBackgroundColor:'#ff3b3b',borderWidth:2}]},options:baseOpts});
    if(histEarChart)histEarChart.destroy();
    const ectx=$('hist-ear-chart');if(ectx)histEarChart=new Chart(ectx,{type:'line',data:{labels:chartLabels,datasets:[{data:chrono.map(r=>r.avg_ear||0),borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.1)',fill:true,tension:.3,pointRadius:3,pointBackgroundColor:'#00d4ff',borderWidth:2}]},options:baseOpts});
  }catch(e){console.error('History error:',e);}
}

function exportCSV(){
  if(!supabaseClient||!currentUser)return;
  supabaseClient.from('sessions').select('*').eq('user_id',currentUser.id).order('created_at',{ascending:false}).then(({data})=>{
    if(!data||!data.length)return alert('No sessions to export');
    const headers=['Date','Label','Mode','Planned(s)','Actual(s)','Alerts','AvgEAR','MinEAR','MaxEpisode(s)'];
    const rows=data.map(r=>[new Date(r.created_at).toISOString(),r.label,r.mode,r.planned_duration||'',r.actual_duration,r.alert_count,r.avg_ear,r.min_ear,r.max_drowsy_episode]);
    const csv=[headers,...rows].map(r=>r.join(',')).join('\n');
    const blob=new Blob([csv],{type:'text/csv'});
    const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`drowsguard_${new Date().toISOString().split('T')[0]}.csv`;a.click();
  });
}

// ===== INIT =====
initSupabase();
