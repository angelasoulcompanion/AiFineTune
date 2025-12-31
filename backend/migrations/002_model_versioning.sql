-- ============================================================
-- AiFineTune Platform - Model Versioning Migration
-- Created: 2025-12-31
-- ============================================================

-- Add versioning columns to finetune_models
ALTER TABLE finetune_models
ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS version_tag VARCHAR(50),
ADD COLUMN IF NOT EXISTS is_latest BOOLEAN DEFAULT TRUE,
ADD COLUMN IF NOT EXISTS version_notes TEXT;

-- Create index for version queries
CREATE INDEX IF NOT EXISTS idx_finetune_models_version ON finetune_models(parent_model_id, version);
CREATE INDEX IF NOT EXISTS idx_finetune_models_latest ON finetune_models(parent_model_id, is_latest) WHERE is_latest = TRUE;

-- Function to auto-increment version for models with same parent
CREATE OR REPLACE FUNCTION auto_increment_model_version()
RETURNS TRIGGER AS $$
DECLARE
    max_version INTEGER;
BEGIN
    -- If this is a fine-tuned model (has parent)
    IF NEW.parent_model_id IS NOT NULL THEN
        -- Get the max version for this parent
        SELECT COALESCE(MAX(version), 0) INTO max_version
        FROM finetune_models
        WHERE parent_model_id = NEW.parent_model_id
          AND model_id != NEW.model_id;

        -- Set version
        NEW.version := max_version + 1;

        -- Set version tag if not provided
        IF NEW.version_tag IS NULL THEN
            NEW.version_tag := 'v' || NEW.version;
        END IF;

        -- Mark previous latest as not latest
        UPDATE finetune_models
        SET is_latest = FALSE
        WHERE parent_model_id = NEW.parent_model_id
          AND is_latest = TRUE
          AND model_id != NEW.model_id;

        -- This is the new latest
        NEW.is_latest := TRUE;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for auto versioning
DROP TRIGGER IF EXISTS trigger_model_version ON finetune_models;
CREATE TRIGGER trigger_model_version
    BEFORE INSERT ON finetune_models
    FOR EACH ROW
    EXECUTE FUNCTION auto_increment_model_version();

-- View for model version history
CREATE OR REPLACE VIEW model_version_history AS
SELECT
    m.model_id,
    m.name,
    m.version,
    m.version_tag,
    m.is_latest,
    m.version_notes,
    m.created_at,
    m.model_type,
    m.status,
    p.name as parent_name,
    p.base_model_id as original_base_model,
    j.training_method,
    j.current_loss as final_loss,
    j.completed_at as trained_at
FROM finetune_models m
LEFT JOIN finetune_models p ON m.parent_model_id = p.model_id
LEFT JOIN finetune_training_jobs j ON m.training_job_id = j.job_id
WHERE m.parent_model_id IS NOT NULL
ORDER BY m.parent_model_id, m.version DESC;

-- ============================================================
-- DONE
-- ============================================================
