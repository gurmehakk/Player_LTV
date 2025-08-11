WITH install_cohort AS (
-- Get install date for Feb-March installs
SELECT 
    COALESCE(gaid, idfa, android_id, custom_user_id) as user_id,
    DATE(MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64))))) as install_date,
    MIN(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) as install_timestamp
FROM {table_path}
WHERE DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) >= '2024-06-01'
    AND DATE(COALESCE(attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(server_timestamp_unix_utc AS INT64)))) <= '2025-08-01'
    AND COALESCE(gaid, idfa, android_id, custom_user_id) IS NOT NULL
    AND COALESCE(gaid, idfa, android_id, custom_user_id) != ''  -- Remove empty user IDs
GROUP BY COALESCE(gaid, idfa, android_id, custom_user_id)
),

user_demographics AS (
-- Basic demographics and attribution info
SELECT 
    ic.user_id,
    ic.install_date,
    ic.install_timestamp,
    
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.country END) as country,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.city END) as city,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.state END) as state,
    
    -- Derive platform
    ANY_VALUE(CASE 
    WHEN e.idfa IS NOT NULL OR e.idfa_md5 IS NOT NULL OR e.idfv IS NOT NULL THEN 'ios'
    WHEN e.gaid IS NOT NULL OR e.android_id IS NOT NULL THEN 'android'
    ELSE 'unknown'
    END) as platform,
    
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.os_version END) as os_version,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.app_version END) as app_version,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.install_source END) as install_source,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.campaign_name END) as campaign_name,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.partner END) as partner,
    ANY_VALUE(CASE WHEN DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date THEN e.publisher_name END) as publisher_name
    
FROM install_cohort ic
LEFT JOIN `{table_path}` e
    ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
    AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) = ic.install_date
GROUP BY ic.user_id, ic.install_date, ic.install_timestamp
),

-- D0-D3 FEATURE WINDOW for training
feature_window_events AS (
SELECT 
    ic.user_id,
    COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64))) as event_timestamp,
    e.session_id,
    e.name,
    e.arguments,
    e.product_name,
    e.product_sku,
    e.product_category,
    e.product_price,
    e.product_quantity,
    COALESCE(e.is_fingerprinted, false) as is_fingerprinted,
    COALESCE(e.is_reengagement, false) as is_reengagement,
    COALESCE(e.is_view_through, false) as is_view_through,
    DATE_DIFF(DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))), ic.install_date, DAY) as day_offset
FROM install_cohort ic
LEFT JOIN {table_path} e
    ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
    AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN ic.install_date AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS} DAY)
WHERE e.name IS NOT NULL
),

-- TARGET: D4-D33 LTV CALCULATION
ltv_target AS (
SELECT 
    ic.user_id,
    SUM(COALESCE(e.converted_revenue, 0)) as ltv_target,
    SUM(COALESCE(e.converted_revenue, 0)) as received_ltv_target,
    COUNTIF(COALESCE(e.converted_revenue, 0) > 0) as purchase_events_target
FROM install_cohort ic
LEFT JOIN {table_path} e
    ON ic.user_id = COALESCE(e.gaid, e.idfa, e.android_id, e.custom_user_id)
    AND DATE(COALESCE(e.attribution_event_timestamp, TIMESTAMP_SECONDS(CAST(e.server_timestamp_unix_utc AS INT64)))) BETWEEN DATE_ADD(ic.install_date, INTERVAL {FEATURE_DAYS_PLUS_1} DAY) AND DATE_ADD(ic.install_date, INTERVAL {FEATURE_PLUS_PREDICTION_DAYS} DAY)
GROUP BY ic.user_id
),

-- FEATURE ENGINEERING (Generic names)
basic_engagement_metrics AS (
SELECT 
    user_id,
    COUNT(*) as total_events,
    COUNT(DISTINCT session_id) as total_sessions,
    COUNT(DISTINCT DATE(event_timestamp)) as active_days,
    
    -- Period-based events (generic names)
    COUNTIF(day_offset = 0) as period_1_events,        -- Day 0 events
    COUNTIF(day_offset = 1) as period_2_events,        -- Day 1 events  
    COUNTIF(day_offset = 2) as period_3_events,        -- Day 2 events
    COUNTIF(day_offset = 3) as period_4_events,        -- Day 3 events
    
    -- Period-based sessions (generic names)
    COUNT(DISTINCT CASE WHEN day_offset = 0 THEN session_id END) as period_1_sessions,
    COUNT(DISTINCT CASE WHEN day_offset = 1 THEN session_id END) as period_2_sessions,
    COUNT(DISTINCT CASE WHEN day_offset = 2 THEN session_id END) as period_3_sessions,
    COUNT(DISTINCT CASE WHEN day_offset = 3 THEN session_id END) as period_4_sessions,
    
    -- Retention flags (generic names)
    CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 1 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as next_period_retained,
    CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 2 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_2_retained,
    CASE WHEN COUNT(DISTINCT CASE WHEN day_offset >= 3 THEN DATE(event_timestamp) END) >= 1 THEN 1 ELSE 0 END as period_plus_3_retained,
    
    MIN(event_timestamp) as first_event_timestamp,
    MAX(event_timestamp) as last_event_timestamp
FROM feature_window_events
WHERE user_id IS NOT NULL
GROUP BY user_id
),

time_pattern_features AS (
SELECT 
    user_id,
    COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 9 AND 17) as business_hours_events,
    COUNTIF(EXTRACT(HOUR FROM event_timestamp) BETWEEN 18 AND 22) as evening_events,
    COUNTIF(EXTRACT(HOUR FROM event_timestamp) >= 23 OR EXTRACT(HOUR FROM event_timestamp) <= 6) as late_night_events,
    COUNTIF(EXTRACT(DAYOFWEEK FROM event_timestamp) IN (1, 7)) as weekend_events,
    COUNT(DISTINCT EXTRACT(HOUR FROM event_timestamp)) as active_hours_spread,
    COUNT(DISTINCT EXTRACT(DAYOFWEEK FROM event_timestamp)) as active_days_of_week,
    COUNTIF(is_fingerprinted = true) as fingerprinted_events,
    COUNTIF(is_reengagement = true) as reengagement_events,
    COUNTIF(is_view_through = true) as view_through_events
FROM feature_window_events
WHERE user_id IS NOT NULL
GROUP BY user_id
),

-- Separate specific event queries instead of LIKE operations
tutorial_events AS (
SELECT 
    user_id,
    COUNT(*) as tutorial_completions,
    MIN(event_timestamp) as tutorial_completion_timestamp
FROM feature_window_events
WHERE user_id IS NOT NULL 
    AND (name IN ('ftue_completed', 'tutorial_completed', 'onboarding_completed', 'tutorial_complete')
        OR name = 'session_start' AND JSON_EXTRACT_SCALAR(arguments, '$.tutorial_completed') = 'true'
        OR name = 'level_end' AND JSON_EXTRACT_SCALAR(arguments, '$.is_tutorial') = 'true')
GROUP BY user_id
),

level_progression_events AS (
SELECT 
    user_id,
    COUNT(*) as levels_completed,
    MAX(SAFE_CAST(COALESCE(
        JSON_EXTRACT_SCALAR(arguments, '$.level_number'),
        JSON_EXTRACT_SCALAR(arguments, '$.level'),
        JSON_EXTRACT_SCALAR(arguments, '$.content_level_number')
    ) AS INT64)) as max_level_reached
FROM feature_window_events
WHERE user_id IS NOT NULL 
    AND (name IN ('level_complete', 'level_completed', 'level_end', 'level_progress', 'level_up')
        OR name = 'session_start' AND JSON_EXTRACT_SCALAR(arguments, '$.last_completed_level') IS NOT NULL)
GROUP BY user_id
),

ad_engagement_events AS (
SELECT 
    user_id,
    COUNT(*) as total_ads_viewed,
    COUNTIF(name IN ('rewarded_ad_watched', 'rewarded_video_completed', 'ad_reward_earned') 
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') = 'rewarded'
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IN ('rewarded_video', 'reward_ad')) as rewarded_ads_viewed,
    COUNTIF(name IN ('interstitial_ad_watched', 'interstitial_completed') 
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') = 'interstitial'
            OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IN ('interstitial', 'interstitial_ad')) as interstitial_ads_viewed
FROM feature_window_events
WHERE user_id IS NOT NULL 
    AND (name IN ('ad_watched', 'ad_completed', 'ad_impression', 'rewarded_ad_watched', 'interstitial_ad_watched', 
                    'rewarded_video_completed', 'interstitial_completed', 'ad_reward_earned', 'ad_started')
        OR JSON_EXTRACT_SCALAR(arguments, '$.ad_type') IS NOT NULL
        OR JSON_EXTRACT_SCALAR(arguments, '$.ad_placement_name') IS NOT NULL)
GROUP BY user_id
),

monetization_events AS (
SELECT 
    user_id,
    COUNTIF(name IN ('store_opened', 'shop_opened', 'iap_store_opened', 'store_view', 'shop_view')) as store_views,
    COUNTIF(name IN ('purchase_intent', 'iap_clicked', 'purchase_started', 'item_selected', 'buy_button_clicked')) as purchase_intents
FROM feature_window_events
WHERE user_id IS NOT NULL 
    AND (name IN ('store_opened', 'shop_opened', 'iap_store_opened', 'store_view', 'shop_view',
                    'purchase_intent', 'iap_clicked', 'purchase_started', 'item_selected', 'buy_button_clicked')
        OR JSON_EXTRACT_SCALAR(arguments, '$.store_type') IS NOT NULL
        OR JSON_EXTRACT_SCALAR(arguments, '$.purchase_intent') = 'true')
GROUP BY user_id
),

currency_and_product_events AS (
SELECT 
    user_id,
    COUNTIF(name = 'currency_earned') as currency_earned_events,
    COUNTIF(name = 'currency_spent') as currency_spent_events,
    COUNTIF(product_name IS NOT NULL AND COALESCE(product_price, 0) = 0) as product_interactions_non_revenue,
    COUNT(DISTINCT CASE WHEN product_name IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_name END) as unique_products_viewed,
    COUNT(DISTINCT CASE WHEN product_sku IS NOT NULL AND COALESCE(product_price, 0) = 0 THEN product_sku END) as unique_skus_viewed
FROM feature_window_events
WHERE user_id IS NOT NULL
GROUP BY user_id
),

social_and_achievement_events AS (
SELECT 
    user_id,
    COUNTIF(name IN ('social_share', 'invite_sent', 'friend_invited', 'social_connect', 'share_completed')) as social_events,
    COUNTIF(name IN ('achievement_earned', 'trophy_earned', 'leaderboard_view', 'milestone_reached', 'badge_earned')) as achievement_events
FROM feature_window_events
WHERE user_id IS NOT NULL 
    AND (name IN ('social_share', 'invite_sent', 'friend_invited', 'social_connect', 'share_completed',
                    'achievement_earned', 'trophy_earned', 'leaderboard_view', 'milestone_reached', 'badge_earned')
        OR JSON_EXTRACT_SCALAR(arguments, '$.social_action') IS NOT NULL
        OR JSON_EXTRACT_SCALAR(arguments, '$.achievement_type') IS NOT NULL)
GROUP BY user_id
),

game_specific_features AS (
SELECT 
    base_users.user_id,
    COALESCE(te.tutorial_completions, 0) as tutorial_completions,
    CASE WHEN COALESCE(te.tutorial_completions, 0) > 0 THEN 1 ELSE 0 END as tutorial_completed_flag,
    te.tutorial_completion_timestamp,
    COALESCE(lpe.levels_completed, 0) as levels_completed,
    COALESCE(lpe.max_level_reached, 0) as max_level_reached,
    COALESCE(aee.total_ads_viewed, 0) as total_ads_viewed,
    COALESCE(aee.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
    COALESCE(aee.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
    COALESCE(me.store_views, 0) as store_views,
    COALESCE(me.purchase_intents, 0) as purchase_intents,
    COALESCE(cape.currency_earned_events, 0) as currency_earned_events,
    COALESCE(cape.currency_spent_events, 0) as currency_spent_events,
    COALESCE(cape.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
    COALESCE(cape.unique_products_viewed, 0) as unique_products_viewed,
    COALESCE(cape.unique_skus_viewed, 0) as unique_skus_viewed,
    COALESCE(saae.social_events, 0) as social_events,
    COALESCE(saae.achievement_events, 0) as achievement_events
FROM (SELECT DISTINCT user_id FROM feature_window_events WHERE user_id IS NOT NULL) base_users
LEFT JOIN tutorial_events te ON base_users.user_id = te.user_id
LEFT JOIN level_progression_events lpe ON base_users.user_id = lpe.user_id
LEFT JOIN ad_engagement_events aee ON base_users.user_id = aee.user_id
LEFT JOIN monetization_events me ON base_users.user_id = me.user_id
LEFT JOIN currency_and_product_events cape ON base_users.user_id = cape.user_id
LEFT JOIN social_and_achievement_events saae ON base_users.user_id = saae.user_id
),

session_timing_features AS (
SELECT 
    user_id,
    AVG(session_duration_minutes) as avg_session_length,
    SUM(session_duration_minutes) as total_playtime_mins,
    MAX(session_duration_minutes) as max_session_length,
    MIN(CASE WHEN session_rank = 1 THEN session_start_hour END) as first_session_hour,
    MIN(CASE WHEN session_rank = 1 THEN session_day_of_week END) as first_session_day_of_week
FROM (
    SELECT 
    user_id,
    session_id,
    MIN(event_timestamp) as session_start,
    MAX(event_timestamp) as session_end,
    EXTRACT(HOUR FROM MIN(event_timestamp)) as session_start_hour,
    EXTRACT(DAYOFWEEK FROM MIN(event_timestamp)) as session_day_of_week,
    TIMESTAMP_DIFF(MAX(event_timestamp), MIN(event_timestamp), MINUTE) as session_duration_minutes,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY MIN(event_timestamp)) as session_rank
    FROM feature_window_events
    WHERE session_id IS NOT NULL AND user_id IS NOT NULL
    GROUP BY user_id, session_id
)
GROUP BY user_id
)

-- FINAL TRAIN DATA OUTPUT
SELECT 
ud.user_id,
ud.install_date,
ud.install_timestamp,

-- Demographics
ud.country,
ud.city,
ud.state,
ud.platform,
ud.os_version,
ud.app_version,
ud.install_source,
ud.campaign_name,
ud.partner,
ud.publisher_name,

-- Basic engagement metrics (generic names)
COALESCE(bem.total_events, 0) as total_events,
COALESCE(bem.total_sessions, 0) as total_sessions,
COALESCE(bem.active_days, 0) as active_days,
COALESCE(bem.period_1_events, 0) as period_1_events,
COALESCE(bem.period_2_events, 0) as period_2_events,
COALESCE(bem.period_3_events, 0) as period_3_events,
COALESCE(bem.period_4_events, 0) as period_4_events,
COALESCE(bem.period_1_sessions, 0) as period_1_sessions,
COALESCE(bem.period_2_sessions, 0) as period_2_sessions,
COALESCE(bem.period_3_sessions, 0) as period_3_sessions,
COALESCE(bem.period_4_sessions, 0) as period_4_sessions,
COALESCE(bem.next_period_retained, 0) as next_period_retained,
COALESCE(bem.period_plus_2_retained, 0) as period_plus_2_retained,
COALESCE(bem.period_plus_3_retained, 0) as period_plus_3_retained,

-- Time patterns
COALESCE(tpf.business_hours_events, 0) as business_hours_events,
COALESCE(tpf.evening_events, 0) as evening_events,
COALESCE(tpf.late_night_events, 0) as late_night_events,
COALESCE(tpf.weekend_events, 0) as weekend_events,
COALESCE(tpf.active_hours_spread, 0) as active_hours_spread,
COALESCE(tpf.active_days_of_week, 0) as active_days_of_week,
COALESCE(tpf.fingerprinted_events, 0) as fingerprinted_events,
COALESCE(tpf.reengagement_events, 0) as reengagement_events,
COALESCE(tpf.view_through_events, 0) as view_through_events,

-- Game features
COALESCE(gsf.tutorial_completed_flag, 0) as tutorial_completed_flag,
gsf.tutorial_completion_timestamp,
COALESCE(gsf.levels_completed, 0) as levels_completed,
COALESCE(gsf.max_level_reached, 0) as max_level_reached,
COALESCE(gsf.total_ads_viewed, 0) as total_ads_viewed,
COALESCE(gsf.rewarded_ads_viewed, 0) as rewarded_ads_viewed,
COALESCE(gsf.interstitial_ads_viewed, 0) as interstitial_ads_viewed,
COALESCE(gsf.store_views, 0) as store_views,
COALESCE(gsf.purchase_intents, 0) as purchase_intents,
COALESCE(gsf.currency_earned_events, 0) as currency_earned_events,
COALESCE(gsf.currency_spent_events, 0) as currency_spent_events,
COALESCE(gsf.product_interactions_non_revenue, 0) as product_interactions_non_revenue,
COALESCE(gsf.unique_products_viewed, 0) as unique_products_viewed,
COALESCE(gsf.unique_skus_viewed, 0) as unique_skus_viewed,
COALESCE(gsf.social_events, 0) as social_events,
COALESCE(gsf.achievement_events, 0) as achievement_events,

-- Session timing
COALESCE(stf.avg_session_length, 0) as avg_session_length,
COALESCE(stf.total_playtime_mins, 0) as total_playtime_mins,
COALESCE(stf.max_session_length, 0) as max_session_length,
stf.first_session_hour,
stf.first_session_day_of_week,

-- Install timing
EXTRACT(HOUR FROM ud.install_timestamp) as install_hour,
EXTRACT(DAYOFWEEK FROM ud.install_timestamp) as install_day_of_week,
EXTRACT(MONTH FROM ud.install_timestamp) as install_month,
TIMESTAMP_DIFF(bem.first_event_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_first_event,
TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) as minutes_install_to_tutorial,

-- Advanced categorical features
CASE 
    WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 0 AND 5 THEN 'late_night_install'
    WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 6 AND 11 THEN 'morning_install' 
    WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 12 AND 17 THEN 'afternoon_install'
    WHEN EXTRACT(HOUR FROM ud.install_timestamp) BETWEEN 18 AND 23 THEN 'evening_install'
END as install_time_segment,

CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 1 THEN 1 ELSE 0 END as install_sunday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 2 THEN 1 ELSE 0 END as install_monday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 3 THEN 1 ELSE 0 END as install_tuesday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 4 THEN 1 ELSE 0 END as install_wednesday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 5 THEN 1 ELSE 0 END as install_thursday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 6 THEN 1 ELSE 0 END as install_friday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) = 7 THEN 1 ELSE 0 END as install_saturday,
CASE WHEN EXTRACT(DAYOFWEEK FROM ud.install_timestamp) IN (1, 7) THEN 1 ELSE 0 END as weekend_install,

-- OS sophistication
CASE 
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 13 THEN 'android_premium'
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 11 THEN 'android_modern'
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'android_standard'
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 7 THEN 'android_legacy'
    WHEN ud.platform = 'android' THEN 'android_ancient'
    WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 16 THEN 'ios_premium'
    WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 14 THEN 'ios_modern'
    WHEN ud.platform = 'ios' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'ios_standard'
    ELSE 'ios_legacy'
END as os_sophistication_tier,

-- Country economic tier
CASE 
    WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') THEN 'tier_1_economy'
    WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES', 'BE', 'AT', 'FI', 'IE', 'NZ') THEN 'tier_1b_economy'
    WHEN ud.country IN ('CN', 'SG', 'HK', 'TW', 'AE', 'QA', 'KW', 'SA', 'BH', 'OM') THEN 'tier_2_economy'
    WHEN ud.country IN ('BR', 'MX', 'AR', 'CL', 'RU', 'TR', 'PL', 'CZ', 'HU', 'GR') THEN 'tier_3_economy'
    WHEN ud.country IN ('IN', 'ID', 'TH', 'MY', 'PH', 'VN', 'ZA', 'EG', 'CO', 'PE') THEN 'tier_4_economy'
    ELSE 'tier_5_economy'
END as economic_tier,

-- Device tier
CASE 
    WHEN ud.platform = 'ios' THEN 'premium_platform'
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 12 THEN 'premium_android'
    WHEN ud.platform = 'android' AND SAFE_CAST(SPLIT(ud.os_version, '.')[OFFSET(0)] AS INT64) >= 9 THEN 'mid_tier_android'
    ELSE 'budget_android'
END as device_tier,

-- Ad behavior
CASE 
    WHEN COALESCE(gsf.rewarded_ads_viewed, 0) >= 5 AND COALESCE(gsf.interstitial_ads_viewed, 0) <= 2 THEN 'reward_seeker'
    WHEN COALESCE(gsf.interstitial_ads_viewed, 0) >= 3 AND COALESCE(gsf.rewarded_ads_viewed, 0) <= 1 THEN 'ad_tolerant'
    WHEN COALESCE(gsf.total_ads_viewed, 0) >= 8 THEN 'high_ad_engagement'
    WHEN COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'moderate_ad_engagement'
    WHEN COALESCE(gsf.total_ads_viewed, 0) >= 1 THEN 'low_ad_engagement'
    ELSE 'ad_avoider'
END as ad_behavior_profile,

-- Tutorial completion speed
CASE 
    WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 5 THEN 'instant_tutorial'
    WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 15 THEN 'fast_tutorial'
    WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 AND TIMESTAMP_DIFF(gsf.tutorial_completion_timestamp, ud.install_timestamp, MINUTE) <= 60 THEN 'slow_tutorial'
    WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 THEN 'very_slow_tutorial'
    ELSE 'no_tutorial'
END as tutorial_completion_speed,

-- Progression tier
CASE 
    WHEN COALESCE(gsf.levels_completed, 0) >= 10 THEN 'fast_progressor'
    WHEN COALESCE(gsf.levels_completed, 0) >= 5 THEN 'moderate_progressor'
    WHEN COALESCE(gsf.levels_completed, 0) >= 2 THEN 'slow_progressor'
    WHEN COALESCE(gsf.levels_completed, 0) >= 1 THEN 'minimal_progressor'
    ELSE 'non_progressor'
END as progression_tier,

-- Calculated ratios (generic names)
SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as avg_events_per_session,
SAFE_DIVIDE(COALESCE(bem.total_events, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_events_per_active_day,
SAFE_DIVIDE(COALESCE(bem.total_sessions, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as avg_sessions_per_day,
SAFE_DIVIDE(COALESCE(tpf.business_hours_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as business_hours_ratio,
SAFE_DIVIDE(COALESCE(tpf.evening_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as evening_activity_ratio,
SAFE_DIVIDE(COALESCE(tpf.weekend_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as weekend_activity_ratio,
SAFE_DIVIDE(COALESCE(tpf.late_night_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as late_night_ratio,
SAFE_DIVIDE(COALESCE(gsf.rewarded_ads_viewed, 0), NULLIF(COALESCE(gsf.total_ads_viewed, 0), 0)) as rewarded_ad_ratio,
SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_sessions, 0), 0)) as ads_per_session,
SAFE_DIVIDE(COALESCE(gsf.total_ads_viewed, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as ad_engagement_rate,
SAFE_DIVIDE(COALESCE(gsf.product_interactions_non_revenue, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as product_interest_ratio,
SAFE_DIVIDE(COALESCE(gsf.store_views, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as store_engagement_ratio,
SAFE_DIVIDE(COALESCE(gsf.purchase_intents, 0), NULLIF(COALESCE(gsf.store_views, 0), 0)) as store_conversion_intent_rate,
SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.active_days, 0), 0)) as levels_per_day,
SAFE_DIVIDE(COALESCE(gsf.levels_completed, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as levels_per_session,

-- Period growth ratios (generic names)
SAFE_DIVIDE(COALESCE(bem.period_2_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_2_to_1_growth_ratio,
SAFE_DIVIDE(COALESCE(bem.period_4_events, 0), NULLIF(COALESCE(bem.period_1_events, 0), 0)) as period_4_to_1_growth_ratio,
SAFE_DIVIDE(COALESCE(bem.period_2_sessions, 0), NULLIF(COALESCE(bem.period_1_sessions, 0), 0)) as session_growth_period_1_to_2,

SAFE_DIVIDE(COALESCE(bem.active_days, 0), 4) as activity_consistency_ratio,
SAFE_DIVIDE(COALESCE(tpf.active_hours_spread, 0), 24) as time_diversity_ratio,
SAFE_DIVIDE(COALESCE(tpf.active_days_of_week, 0), 7) as weekly_consistency_ratio,
SAFE_DIVIDE(COALESCE(tpf.fingerprinted_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as fingerprinted_ratio,
SAFE_DIVIDE(COALESCE(tpf.reengagement_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as reengagement_ratio,
SAFE_DIVIDE(COALESCE(tpf.view_through_events, 0), NULLIF(COALESCE(bem.total_events, 0), 0)) as view_through_ratio,

-- Platform × Country combo
CONCAT(
    CASE WHEN ud.platform = 'ios' THEN 'ios' ELSE 'android' END,
    '_',
    CASE 
    WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU') THEN 'premium_geo'
    WHEN ud.country IN ('JP', 'KR', 'FR', 'IT', 'ES') THEN 'good_geo'
    WHEN ud.country IN ('CN', 'BR', 'RU', 'IN', 'MX') THEN 'medium_geo'
    ELSE 'other_geo'
    END
) as platform_geo_combo,

-- Monetization potential
CASE 
    WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'NL', 'CH', 'NO', 'SE', 'DK') 
        AND ud.platform = 'ios' 
        AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
        AND COALESCE(gsf.total_ads_viewed, 0) >= 3 THEN 'high_monetization_potential'
    WHEN ud.country IN ('US', 'GB', 'DE', 'CA', 'AU', 'JP', 'KR', 'FR', 'IT', 'ES') 
        AND COALESCE(gsf.tutorial_completed_flag, 0) = 1 
        AND COALESCE(bem.period_1_sessions, 0) >= 3 THEN 'medium_monetization_potential'
    WHEN COALESCE(gsf.tutorial_completed_flag, 0) = 1 OR COALESCE(bem.period_1_sessions, 0) >= 2 THEN 'low_monetization_potential'
    ELSE 'minimal_monetization_potential'
END as monetization_potential_tier,

-- ==========================================
-- TARGET VARIABLES (D4-D33 LTV)
-- ==========================================
COALESCE(lt.ltv_target, 0) as ltv_target,
COALESCE(lt.received_ltv_target, 0) as received_ltv_target,
COALESCE(lt.purchase_events_target, 0) as purchase_events_target

FROM user_demographics ud
LEFT JOIN basic_engagement_metrics bem ON ud.user_id = bem.user_id
LEFT JOIN time_pattern_features tpf ON ud.user_id = tpf.user_id
LEFT JOIN game_specific_features gsf ON ud.user_id = gsf.user_id
LEFT JOIN session_timing_features stf ON ud.user_id = stf.user_id
LEFT JOIN ltv_target lt ON ud.user_id = lt.user_id
WHERE COALESCE(bem.total_events, 0) >= 1  -- At least some activity
ORDER BY ud.install_date, ud.user_id;

    
