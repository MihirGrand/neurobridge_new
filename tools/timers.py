import datetime
import json
import os
import threading
import time
from typing import List, Dict

TIMERS_FILE = os.path.join(os.path.dirname(__file__), "timers.json")

def _load_timers() -> List[Dict]:
    if not os.path.exists(TIMERS_FILE):
        return []
    try:
        with open(TIMERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def _save_timers(timers: List[Dict]):
    with open(TIMERS_FILE, "w", encoding="utf-8") as f:
        json.dump(timers, f, indent=4)

def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.datetime.now().strftime("%I:%M %p, %A, %B %d, %Y")

def set_timer(minutes: float, label: str) -> str:
    """Sets a new timer for a given number of minutes with a label.
    
    Args:
        minutes: The duration of the timer in minutes.
        label: A short description of the timer (e.g., 'Pasta', 'Laundry').
        
    Returns:
        A confirmation message.
    """
    timers = _load_timers()
    end_time = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    
    timer = {
        "id": str(int(time.time() * 1000)),
        "label": label,
        "end_time": end_time.isoformat(),
        "notified": False
    }
    
    timers.append(timer)
    _save_timers(timers)
    
    return f"Timer '{label}' set for {minutes} minutes! It will go off at {end_time.strftime('%I:%M %p')}."

def get_timers() -> str:
    """Gets information about all currently active timers.
    
    Returns:
        A formatted string listing all active timers and their remaining time.
    """
    timers = _load_timers()
    now = datetime.datetime.now()
    active_timers = [t for t in timers if not t.get("notified")]
    
    if not active_timers:
        return "There are no active timers."
        
    result = ["Active timers:"]
    for t in active_timers:
        end_time = datetime.datetime.fromisoformat(t["end_time"])
        if end_time > now:
            remaining = end_time - now
            mins, secs = divmod(int(remaining.total_seconds()), 60)
            result.append(f"- '{t['label']}' with {mins} minutes and {secs} seconds remaining.")
        else:
            result.append(f"- '{t['label']}' (already elapsed).")
            
    return "\n".join(result)

def start_timer_watcher(bot_instance):
    """Starts a background thread that monitors timers and pushes alerts to the chat."""
    def watcher_loop():
        while True:
            timers = _load_timers()
            now = datetime.datetime.now()
            updated = False
            
            for t in timers:
                if not t.get("notified"):
                    end_time = datetime.datetime.fromisoformat(t["end_time"])
                    if now >= end_time:
                        t["notified"] = True
                        updated = True
                        label = t["label"]
                        # Push an alert directly into the chat stream!
                        msg = f"SYSTEM ALERT: Timer '{label}' has elapsed right now! Please notify the user prominently."
                        
                        # Use a new thread to not block the watcher
                        def alert():
                            bot_instance.chat(msg)
                            
                        threading.Thread(target=alert, daemon=True).start()
            
            if updated:
                # Remove completed timers to keep the list clean over time
                timers = [t for t in timers if not t.get("notified")]
                _save_timers(timers)
                
            time.sleep(1)
            
    t = threading.Thread(target=watcher_loop, daemon=True)
    t.start()
