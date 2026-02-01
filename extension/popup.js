/**
 * FactPulse Popup Script v2.0
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
    verdictSummary: document.getElementById('verdict-summary'),
    countTrue: document.getElementById('count-true'),
    countFalse: document.getElementById('count-false'),
    countPartial: document.getElementById('count-partial'),
    countUnknown: document.getElementById('count-unknown'),
    btnAnalyze: document.getElementById('btn-analyze'),
    btnClear: document.getElementById('btn-clear'),
    btnSettings: document.getElementById('btn-settings'),
    loading: document.getElementById('loading')
  };

  // ============================================================================
  // STATE
  // ============================================================================
  
  let isConnected = false;
  let lastResults = null;

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
   * Met à jour le résumé des verdicts.
   */
  function updateVerdictSummary(results) {
    if (!results || results.length === 0) {
      elements.verdictSummary.style.display = 'none';
      return;
    }
    
    const counts = {
      TRUE: 0,
      FALSE: 0,
      PARTIALLY_TRUE: 0,
      NOT_VERIFIABLE: 0
    };
    
    for (const r of results) {
      if (counts.hasOwnProperty(r.verdict)) {
        counts[r.verdict]++;
      }
    }
    
    elements.countTrue.textContent = counts.TRUE;
    elements.countFalse.textContent = counts.FALSE;
    elements.countPartial.textContent = counts.PARTIALLY_TRUE;
    elements.countUnknown.textContent = counts.NOT_VERIFIABLE;
    
    elements.verdictSummary.style.display = 'flex';
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
   * Lance l'analyse de la page courante (viewport uniquement).
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
        
        // Récupérer les détails pour le résumé des verdicts
        const statusResponse = await chrome.tabs.sendMessage(tab.id, { action: 'getStatus' });
        if (statusResponse && statusResponse.results && statusResponse.results.results) {
          updateVerdictSummary(statusResponse.results.results);
        }
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
   * Efface les annotations de la page.
   */
  async function clearAnnotations() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab || !tab.id) {
        return;
      }
      
      await chrome.tabs.sendMessage(tab.id, { action: 'clear' });
      
      // Réinitialiser l'affichage
      updateStats(null, null);
      elements.verdictSummary.style.display = 'none';
      
    } catch (error) {
      console.error('Clear error:', error);
    }
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
    
    // Charger l'état depuis le content script
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab && tab.id) {
        const statusResponse = await chrome.tabs.sendMessage(tab.id, { action: 'getStatus' });
        if (statusResponse && statusResponse.results) {
          const results = statusResponse.results;
          if (results.claims_verified !== undefined) {
            updateStats(results.claims_verified, results.timing?.total_ms);
          }
          if (results.results) {
            updateVerdictSummary(results.results);
          }
        }
      }
    } catch (e) {
      // Content script pas encore injecté, c'est OK
      console.log('Content script not available yet');
    }
    
    // Event listeners
    elements.btnAnalyze.addEventListener('click', analyzePage);
    elements.btnClear.addEventListener('click', clearAnnotations);
    elements.btnSettings.addEventListener('click', openSettings);
  }
  
  // Démarrer
  init();

})();
