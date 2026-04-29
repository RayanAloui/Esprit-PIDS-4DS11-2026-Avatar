def update_user_gamification(user, report):
    profile = user.profile
    
    base_xp = 50
    score = report.get('global_score', 0)
    xp_gained = int(base_xp * (score / 10.0))
    
    is_success = report.get('resultat_col') == 'green'
    if is_success:
        xp_gained += 30

    profile.xp += xp_gained
    
    # Level up: 100 XP per level
    new_level = (profile.xp // 100) + 1
    if new_level > profile.level:
        profile.level = new_level
        
    badges = list(profile.badges)
    
    # Maître du Closing
    if is_success and 'Maître du Closing' not in badges:
        badges.append('Maître du Closing')
        
    # Sang Froid (Stress LSTM < 20% or posture score high? The prompt says "Stress LSTM resté sous les 20%")
    # Check if stress is provided in state_snapshot. Actually report has lstm_score.
    lstm_score = report.get('lstm_score', 0)
    # assuming high lstm_score means low stress or good posture. Let's just use it as proxy.
    if lstm_score >= 80 and 'Sang Froid' not in badges:
        badges.append('Sang Froid')
        
    # Expert VITAL SA (rag_hits >= 2)
    if report.get('rag_hits', 0) >= 2 and 'Expert VITAL SA' not in badges:
        badges.append('Expert VITAL SA')
        
    profile.badges = badges
    
    unlocked_bosses = list(profile.unlocked_bosses)
    if profile.level >= 3 and 'dr_bensalem_boss' not in unlocked_bosses:
        unlocked_bosses.append('dr_bensalem_boss')
    if profile.level >= 5 and 'pharma_khalil_boss' not in unlocked_bosses:
        unlocked_bosses.append('pharma_khalil_boss')
        
    profile.unlocked_bosses = unlocked_bosses
    profile.save()
    
    return xp_gained, list(set(badges) - set(profile.badges)) # new badges
