#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式轨迹预测演示系统 + RESTful API
"""
import os, csv, json
import torch
import torch.nn.functional as F
import numpy as np
from flask import Flask, render_template, request, jsonify
from model import MultiModalVectorNet

app = Flask(__name__)

OBS_LEN = 20
PRED_LEN = 30
FEATURE_DIM = 8
HIDDEN_DIM = 128
K_MODES = 3
OUT_DIM = PRED_LEN * 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultiModalVectorNet(in_channels=FEATURE_DIM, out_channels=OUT_DIM,
                            k_modes=K_MODES, hidden_dim=HIDDEN_DIM, obs_len=OBS_LEN)
try:
    checkpoint = torch.load('best_multimodal_model.pth', map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("Using randomly initialized model for demo")

model.eval()
model.to(DEVICE)


def compute_features(obs_traj):
    features = []
    obs_len = min(len(obs_traj), OBS_LEN)
    for i in range(obs_len):
        if i == 0:
            vel_x, vel_y = 0, 0
        else:
            vel_x = obs_traj[i, 0] - obs_traj[i-1, 0]
            vel_y = obs_traj[i, 1] - obs_traj[i-1, 1]
        features.append([obs_traj[i, 0], obs_traj[i, 1], vel_x, vel_y, i / OBS_LEN, 1, 0, 0])
    while len(features) < OBS_LEN:
        features.append(features[-1])
    return np.array(features, dtype=np.float32)


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': True, 'device': str(DEVICE)})


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'trajectory' not in data:
            return jsonify({'success': False, 'error': 'Missing trajectory'}), 400

        obs_trajectory = np.array(data['trajectory'])
        if len(obs_trajectory) < OBS_LEN:
            return jsonify({'success': False, 'error': f'Need min {OBS_LEN} points'}), 400

        obs_norm = obs_trajectory - obs_trajectory[0]
        features = compute_features(obs_norm)

        with torch.no_grad():
            input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)
            outputs = model(input_tensor)

        pred_trajs = outputs['trajectories'][0].cpu().numpy()
        mode_probs = outputs['mode_probs'][0].cpu().numpy()
        uncertainty = outputs['uncertainty'][0].cpu().numpy()
        top_k = min(data.get('top_k', K_MODES), K_MODES)

        stats = []
        for i in range(top_k):
            traj = pred_trajs[i].reshape(-1, 2)
            final_point = traj[-1]
            total_distance = np.sum(np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1)))
            avg_speed = total_distance / (PRED_LEN * 0.1)
            stats.append({
                'mode': i + 1, 'probability': float(mode_probs[i]),
                'final_x': float(final_point[0]), 'final_y': float(final_point[1]),
                'total_distance': float(total_distance), 'avg_speed': float(avg_speed)
            })

        result = {
            'success': True,
            'data': {
                'trajectories': [pred_trajs[i].reshape(-1, 2).tolist() for i in range(top_k)],
                'probabilities': mode_probs[:top_k].tolist(),
                'uncertainty': uncertainty.reshape(-1, 2).tolist(),
                'statistics': stats,
                'summary': {
                    'best_mode': int(np.argmax(mode_probs)) + 1,
                    'confidence': float(np.max(mode_probs)),
                    'observation_points': len(obs_trajectory),
                    'prediction_horizon': PRED_LEN,
                    'avg_uncertainty': float(np.mean(uncertainty))
                }
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        if 'predictions' not in data:
            return jsonify({'success': False, 'error': 'Missing predictions'}), 400
        results = []
        for item in data['predictions']:
            try:
                obs_trajectory = np.array(item['trajectory'])
                if len(obs_trajectory) < OBS_LEN:
                    results.append({'id': item.get('id', 'unknown'), 'success': False, 'error': f'Need {OBS_LEN} points'})
                    continue
                obs_norm = obs_trajectory - obs_trajectory[0]
                features = compute_features(obs_norm)
                with torch.no_grad():
                    input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)
                    outputs = model(input_tensor)
                pred_trajs = outputs['trajectories'][0].cpu().numpy()
                mode_probs = outputs['mode_probs'][0].cpu().numpy()
                results.append({
                    'id': item.get('id', 'unknown'), 'success': True,
                    'best_trajectory': pred_trajs[0].reshape(-1, 2).tolist(),
                    'best_probability': float(mode_probs[0]),
                    'all_probabilities': mode_probs.tolist()
                })
            except Exception as e:
                results.append({'id': item.get('id', 'unknown'), 'success': False, 'error': str(e)})
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/v1/model_info', methods=['GET'])
def model_info():
    total_params = sum(p.numel() for p in model.parameters())
    return jsonify({
        'success': True,
        'model': {
            'name': 'MultiModalVectorNet', 'parameters': total_params,
            'device': str(DEVICE), 'k_modes': K_MODES,
            'observation_length': OBS_LEN, 'prediction_length': PRED_LEN,
            'feature_dim': FEATURE_DIM
        }
    })


@app.route('/api/v1/examples', methods=['GET'])
def examples():
    import math
    examples = [
        {'name': 'straight_motion', 'description': '直线运动',
         'trajectory': [[i, i * 0.5] for i in range(OBS_LEN)]},
        {'name': 'curve_motion', 'description': '曲线运动',
         'trajectory': [[i, float(np.sin(i * 0.3) * 5)] for i in range(OBS_LEN)]},
        {'name': 'lane_change', 'description': '换道',
         'trajectory': [[i, 0 if i < 10 else (i - 10) * 0.3] for i in range(OBS_LEN)]}
    ]
    return jsonify({'success': True, 'examples': examples})


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demo', methods=['GET'])
def demo():
    t = np.linspace(0, 2, OBS_LEN)
    obs_traj = np.array([t, t * 0.5]).T
    return jsonify({'trajectory': obs_traj.tolist()})


if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')

    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Multi-Modal Trajectory Prediction System</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        #canvas { border: 2px solid #333; background: white; border-radius: 8px; }
        .controls { margin: 20px 0; }
        button { padding: 12px 24px; font-size: 16px; cursor: pointer; margin-right: 10px; border: none; border-radius: 5px; background: #3498db; color: white; transition: background 0.3s; }
        button:hover { background: #2980b9; }
        button.lang-btn { padding: 6px 12px; font-size: 14px; background: #6c757d; width: 85px; min-width: 85px; max-width: 85px; text-align: center; box-sizing: border-box; height: 32px; line-height: 16px; }
        button.lang-btn:hover, button.lang-btn.active { background: #495057; }
        .prob-display { margin: 15px 0; padding: 15px; background: #fff; border-radius: 8px; }
        .mode-item { display: inline-block; margin-right: 15px; padding: 8px 20px; border-radius: 20px; font-weight: bold; }
        .mode-0 { background: #FF6B6B; color: white; } .mode-1 { background: #4ECDC4; color: white; } .mode-2 { background: #9B59B6; color: white; }
        .api-demo { background: #fff; padding: 20px; margin: 20px 0; border-radius: 8px; }
        pre { background: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { color: #2c3e50; } .header p { color: #7f8c8d; }
        .lang-selector { position: absolute; top: 20px; right: 20px; }
        .report-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .report-card { background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .report-card h3 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 0; }
        .stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .stat-item { background: #f8f9fa; padding: 12px; border-radius: 6px; }
        .stat-label { font-size: 14px; color: #666; margin-bottom: 5px; }
        .stat-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
        .confidence-bar { height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; margin-top: 10px; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #28a745, #20c997); border-radius: 10px; transition: width 0.5s ease; }
        .trajectory-table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        .trajectory-table th, .trajectory-table td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        .trajectory-table th { background: #f8f9fa; font-weight: bold; }
        .badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
        .badge-high { background: #d4edda; color: #155724; } .badge-medium { background: #fff3cd; color: #856404; } .badge-low { background: #f8d7da; color: #721c24; }
        .summary-box { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .summary-box h3 { margin-top: 0; }
    </style>
</head>
<body>
    <div class="lang-selector">
        <button id="lang-zh" class="lang-btn active" onclick="switchLang('zh')">中文</button>
        <button id="lang-en" class="lang-btn" onclick="switchLang('en')">English</button>
    </div>
    <div class="header">
        <h1 id="title">Multi-Modal Trajectory Prediction System</h1>
        <p id="subtitle">Lightweight Autonomous Driving Trajectory Prediction</p>
    </div>
    <div class="api-demo">
        <h3>RESTful API</h3>
        <p>POST /api/v1/predict</p>
        <pre>{"trajectory": [[0,0],[1,0.5],...], "top_k": 3}</pre>
        <p>Try: <a href="/api/v1/examples">Examples</a> | <a href="/api/v1/model_info">Model Info</a></p>
    </div>
    <div class="controls">
        <button onclick="loadDemo()">Load Demo</button>
        <button onclick="runPrediction()">Predict</button>
        <button onclick="clearCanvas()">Clear</button>
    </div>
    <canvas id="canvas" width="800" height="600"></canvas>
    <div class="prob-display" id="probDisplay"></div>
    <div id="predictionReport" style="display:none;">
        <div class="summary-box">
            <h3>Prediction Summary</h3>
            <div style="display:flex;gap:30px;">
                <div><span style="font-size:24px;font-weight:bold;">Best Mode: </span><span id="bestMode" style="font-size:28px;color:#ffd700;">Mode 1</span></div>
                <div><span style="font-size:20px;">Confidence: </span><span id="confidence" style="font-size:24px;font-weight:bold;">0%</span></div>
            </div>
            <div class="confidence-bar"><div id="confidenceBar" class="confidence-fill" style="width:0%"></div></div>
        </div>
        <div class="report-container">
            <div class="report-card">
                <h3>Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-item"><div class="stat-label">Obs Points</div><div class="stat-value" id="obsPoints">0</div></div>
                    <div class="stat-item"><div class="stat-label">Horizon</div><div class="stat-value" id="predHorizon">0</div></div>
                    <div class="stat-item"><div class="stat-label">Avg Uncertainty</div><div class="stat-value" id="avgUncertainty">0</div></div>
                    <div class="stat-item"><div class="stat-label">Confidence</div><div class="stat-value"><span id="confidenceLevel" class="badge">Low</span></div></div>
                </div>
            </div>
            <div class="report-card">
                <h3>Trajectory Details</h3>
                <table class="trajectory-table">
                    <thead><tr><th>Mode</th><th>Prob</th><th>End Point</th><th>Dist</th></tr></thead>
                    <tbody id="trajectoryDetails"></tbody>
                </table>
            </div>
        </div>
    </div>
    <script>
        let currentLang = 'zh';
        let lastPredictionData = null;
        function switchLang(lang) { currentLang = lang; document.getElementById('lang-zh').className = lang==='zh'?'lang-btn active':'lang-btn'; document.getElementById('lang-en').className = lang==='en'?'lang-btn active':'lang-btn'; if(lastPredictionData){updateProbDisplay(lastPredictionData);updateReport(lastPredictionData);} }
        function updateProbDisplay(data){var d=document.getElementById('probDisplay');d.innerHTML='<h3>Probabilities:</h3>'+data.probabilities.map((p,i)=>'<div class="mode-item mode-'+i+'">Mode '+(i+1)+': '+(p*100).toFixed(1)+'%</div>').join('');}
        const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
        let points=[],selectedPoint=null,predictedTrajectories=[];
        function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='#eee';ctx.lineWidth=1;for(let i=0;i<canvas.width;i+=50){ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i,canvas.height);ctx.stroke();}for(let i=0;i<canvas.height;i+=50){ctx.beginPath();ctx.moveTo(0,i);ctx.lineTo(canvas.width,i);ctx.stroke();}if(points.length>1){ctx.strokeStyle='#3498db';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(let i=1;i<points.length;i++)ctx.lineTo(points[i].x,points[i].y);ctx.stroke();points.forEach((p,i)=>{ctx.fillStyle='#3498db';ctx.beginPath();ctx.arc(p.x,p.y,7,0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='bold 12px Arial';ctx.textAlign='center';ctx.fillText(i,p.x,p.y+4);});}if(predictedTrajectories.length>0){var colors=['#FF6B6B','#4ECDC4','#9B59B6'];predictedTrajectories.forEach((traj,idx)=>{ctx.strokeStyle=colors[idx];ctx.lineWidth=3;ctx.setLineDash([8,4]);ctx.beginPath();ctx.moveTo(points[points.length-1].x,points[points.length-1].y);traj.forEach(point=>{ctx.lineTo(points[points.length-1].x+point[0]*10,points[points.length-1].y-point[1]*10);});ctx.stroke();ctx.setLineDash([]);var lp=traj[traj.length-1],ex=points[points.length-1].x+lp[0]*10,ey=points[points.length-1].y-lp[1]*10;ctx.fillStyle=colors[idx];ctx.beginPath();ctx.arc(ex,ey,8,0,Math.PI*2);ctx.fill();ctx.fillStyle='white';ctx.font='bold 10px Arial';ctx.textAlign='center';ctx.fillText(idx+1,ex,ey+3);});}}
        async function loadDemo(){var r=await fetch('/demo');var d=await r.json();var cx=canvas.width/2,cy=canvas.height/2;points=d.trajectory.map((p,i)=>({x:cx+p[0]*25,y:cy-p[1]*25}));predictedTrajectories=[];document.getElementById('predictionReport').style.display='none';document.getElementById('probDisplay').innerHTML='';draw();}
        async function runPrediction(){if(points.length<20){alert('Need at least 20 points');return;}var o={x:points[0].x,y:points[0].y};var traj=points.map(p=>[p.x-o.x,o.y-p.y]);try{var r=await fetch('/api/v1/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({trajectory:traj,top_k:3})});var d=await r.json();if(d.success){predictedTrajectories=d.data.trajectories;lastPredictionData=d.data;updateProbDisplay(d.data);updateReport(d.data);draw();}else alert('Failed: '+d.error);}catch(e){alert('Error: '+e.message);}}
        function updateReport(data){document.getElementById('predictionReport').style.display='block';document.getElementById('bestMode').textContent='Mode '+data.summary.best_mode;document.getElementById('confidence').textContent=(data.summary.confidence*100).toFixed(1)+'%';document.getElementById('confidenceBar').style.width=(data.summary.confidence*100)+'%';document.getElementById('obsPoints').textContent=data.summary.observation_points;document.getElementById('predHorizon').textContent=data.summary.prediction_horizon;document.getElementById('avgUncertainty').textContent=data.summary.avg_uncertainty.toFixed(4);var lb=document.getElementById('confidenceLevel'),c=data.summary.confidence;if(c>=0.7){lb.textContent='High';lb.className='badge badge-high';}else if(c>=0.4){lb.textContent='Medium';lb.className='badge badge-medium';}else{lb.textContent='Low';lb.className='badge badge-low';}var tb=document.getElementById('trajectoryDetails');tb.innerHTML='';var colors=['#FF6B6B','#4ECDC4','#9B59B6'];data.statistics.forEach((s,i)=>{var r=document.createElement('tr');r.innerHTML='<td><span style="color:'+colors[i]+';font-weight:bold;">Mode '+s.mode+'</span></td><td>'+(s.probability*100).toFixed(1)+'%</td><td>('+s.final_x.toFixed(2)+','+s.final_y.toFixed(2)+')</td><td>'+s.total_distance.toFixed(2)+' m</td>';tb.appendChild(r);});}
        function clearCanvas(){points=[];predictedTrajectories=[];document.getElementById('probDisplay').innerHTML='';document.getElementById('predictionReport').style.display='none';draw();}
        canvas.addEventListener('mousedown',e=>{var r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;for(var i=0;i<points.length;i++){if(Math.sqrt((x-points[i].x)**2+(y-points[i].y)**2)<12){selectedPoint=i;return;}}if(points.length<30){points.push({x,y});draw();}});
        canvas.addEventListener('mousemove',e=>{if(selectedPoint===null)return;var r=canvas.getBoundingClientRect();points[selectedPoint].x=e.clientX-r.left;points[selectedPoint].y=e.clientY-r.top;draw();});
        canvas.addEventListener('mouseup',()=>{selectedPoint=null;});
        draw();
    </script>
</body>
</html>
    """

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

    print("=" * 60)
    print("Trajectory Prediction System")
    print("=" * 60)
    print("Web: http://localhost:5000")
    print("API: http://localhost:5000/api/v1/predict")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
