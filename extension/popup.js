/**
 * FactPulse Popup Script
 * 
 * Gère l'interface popup de l'extension.
 * Communique avec le content script et le backend.
 */

(function() {
  'use strict';

  // ============================================================================
  // CONFIGURATION
  // ============================================================================
  
  const API_URL = 'http://localhost:8000';
  
  // ============================================================================
  // DOM ELEMENTS
  // ============================================================================
  
  const elements = {
    statusIndicator: document.getElementById('status-indicator'),
    statusLabel: document.getElementById('status-label'),
    statusDetail: document.getElementById('status-detail'),
    statClaims: document.getElementById('stat-claims'),
    statTime: document.getElementById('stat-time'),
    btnAnalyze: document.getElementById('btn-analyze'),
    btnSettings: document.getElementById('btn-settings'),
    loading: document.getElementById('loading')
  };

  // ============================================================================
  // STATE
  // ============================================================================
  
  let isConnected = false;
  let stats = {
    totalClaims: 0,
    avgTime: 0
  };

  // ============================================================================
  // API
  // ============================================================================
  
  /**
   * Vérifie la connexion au serveur.
   */
  async function checkServerHealth() {
    try {
      const response = await fetch(`${API_URL}/health`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          connected: true,
          gpuAvailable: data.gpu_available,
          modelsLoaded: data.models_loaded
        };
      }
    } catch (error) {
      console.error('Health check failed:', error);
    }
    
    return { connected: false };
  }
  
  /**
   * Récupère les stats du serveur.
   */
  async function getServerStats() {
    try {
      const response = await fetch(`${API_URL}/stats`, {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error('Stats fetch failed:', error);
    }
    
    return null;
  }

  // ============================================================================
  // UI UPDATES
  // ============================================================================
  
  /**
   * Met à jour l'indicateur de statut.
   */
  function updateStatus(connected, detail = '') {
    isConnected = connected;
    
    if (connected) {
      elements.statusIndicator.className = 'status-indicator connected';
      elements.statusLabel.textContent = 'Connecté';
      elements.statusDetail.textContent = detail || 'Serveur opérationnel';
      elements.btnAnalyze.disabled = false;
    } else {
      elements.statusIndicator.className = 'status-indicator disconnected';
      elements.statusLabel.textContent = 'Déconnecté';
      elements.statusDetail.textContent = detail || 'Serveur non disponible';
      elements.btnAnalyze.disabled = true;
    }
  }
  
  /**
   * Met à jour les statistiques.
   */
  function updateStats(claims, time) {
    elements.statClaims.textContent = claims || '-';
    elements.statTime.textContent = time ? `${time}ms` : '-';
  }
  
  /**
   * Affiche/masque le loading.
   */
  function setLoading(loading) {
    if (loading) {
      elements.loading.classList.add('active');
      elements.btnAnalyze.disabled = true;
    } else {
      elements.loading.classList.remove('active');
      elements.btnAnalyze.disabled = !isConnected;
    }
  }

  // ============================================================================
  // ACTIONS
  // ============================================================================
  
  /**
   * Lance l'analyse de la page courante.
   */
  async function analyzePage() {
    setLoading(true);
    
    try {
      // Récupérer l'onglet actif
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab || !tab.id) {
        throw new Error('Onglet non disponible');
      }
      
      // Envoyer un message au content script
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'analyze' });
      
      if (response && response.success) {
        updateStats(response.claimsCount, response.time);
      } else {
        console.error('Analysis failed:', response);
      }
      
    } catch (error) {
      console.error('Analysis error:', error);
      
      // Fallback: essayer d'injecter et exécuter le content script
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          func: () => {
            if (window.factpulse && window.factpulse.analyze) {
              window.factpulse.analyze();
              return true;
            }
            return false;
          }
        });
      } catch (e) {
        console.error('Script injection failed:', e);
      }
    }
    
    setLoading(false);
  }
  
  /**
   * Ouvre les paramètres.
   */
  function openSettings() {
    // Pour l'instant, ouvre la doc API
    chrome.tabs.create({ url: `${API_URL}/docs` });
  }

  // ============================================================================
  // INITIALIZATION
  // ============================================================================
  
  /**
   * Initialise le popup.
   */
  async function init() {
    // Vérifier la connexion au serveur
    const health = await checkServerHealth();
    
    if (health.connected) {
      let detail = 'Serveur opérationnel';
      if (health.gpuAvailable) {
        detail += ' (GPU ✓)';
      }
      if (health.modelsLoaded) {
        detail += ' - Modèles chargés';
      }
      updateStatus(true, detail);
    } else {
      updateStatus(false, 'Démarrez le serveur: uvicorn backend.api:app');
    }
    
    // Charger les stats depuis le storage
    try {
      const stored = await chrome.storage.local.get(['stats']);
      if (stored.stats) {
        updateStats(stored.stats.totalClaims, stored.stats.avgTime);
      }
    } catch (e) {
      console.error('Storage error:', e);
    }
    
    // Event listeners
    elements.btnAnalyze.addEventListener('click', analyzePage);
    elements.btnSettings.addEventListener('click', openSettings);
  }
  
  // Démarrer
  init();

})();
