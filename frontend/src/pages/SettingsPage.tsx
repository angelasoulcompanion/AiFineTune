import { useState } from 'react';
import {
  Settings,
  Key,
  User,
  Lock,
  CheckCircle,
  AlertCircle,
  Loader2,
  Eye,
  EyeOff,
  ExternalLink,
  Trash2,
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import {
  useHFTokenStatus,
  useSaveHFToken,
  useDeleteHFToken,
  useValidateHFToken,
  useChangePassword,
  useUpdateProfile,
} from '../hooks/useSettings';
import type { HFTokenValidation } from '../hooks/useSettings';

export default function SettingsPage() {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <Settings className="w-8 h-8 text-primary-500" />
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Settings</h1>
      </div>

      {/* Profile Section */}
      <ProfileSection />

      {/* HuggingFace Token Section */}
      <HFTokenSection />

      {/* Password Section */}
      <PasswordSection />
    </div>
  );
}

function ProfileSection() {
  const { user } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [username, setUsername] = useState(user?.username || '');
  const updateProfile = useUpdateProfile();

  const handleSave = async () => {
    try {
      await updateProfile.mutateAsync({ username });
      setIsEditing(false);
    } catch (error: any) {
      alert(error.response?.data?.detail || 'Failed to update profile');
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center gap-3 mb-4">
        <User className="w-5 h-5 text-gray-500" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Profile</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Email
          </label>
          <input
            type="email"
            value={user?.email || ''}
            disabled
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Username
          </label>
          {isEditing ? (
            <div className="flex gap-2">
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
              />
              <button
                onClick={handleSave}
                disabled={updateProfile.isPending}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50"
              >
                {updateProfile.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save'}
              </button>
              <button
                onClick={() => {
                  setUsername(user?.username || '');
                  setIsEditing(false);
                }}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="flex gap-2">
              <input
                type="text"
                value={user?.username || ''}
                disabled
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
              <button
                onClick={() => setIsEditing(true)}
                className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Edit
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function HFTokenSection() {
  const [token, setToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [validationResult, setValidationResult] = useState<HFTokenValidation | null>(null);

  const { data: tokenStatus, isLoading } = useHFTokenStatus();
  const saveToken = useSaveHFToken();
  const deleteToken = useDeleteHFToken();
  const validateToken = useValidateHFToken();

  const handleSaveToken = async () => {
    if (!token.trim()) return;

    try {
      const result = await saveToken.mutateAsync({ hf_token: token, validate: true });
      if (result.is_valid) {
        setToken('');
        setValidationResult({
          valid: true,
          username: result.hf_username,
          name: result.hf_name,
          email: null,
          orgs: [],
          can_write: result.can_write || false,
          error: null,
        });
      }
    } catch (error: any) {
      setValidationResult({
        valid: false,
        username: null,
        name: null,
        email: null,
        orgs: [],
        can_write: false,
        error: error.response?.data?.detail || 'Failed to save token',
      });
    }
  };

  const handleValidate = async () => {
    try {
      const result = await validateToken.mutateAsync();
      setValidationResult(result);
    } catch (error: any) {
      setValidationResult({
        valid: false,
        username: null,
        name: null,
        email: null,
        orgs: [],
        can_write: false,
        error: error.response?.data?.detail || 'Failed to validate token',
      });
    }
  };

  const handleDelete = async () => {
    if (confirm('Are you sure you want to delete your HuggingFace token?')) {
      await deleteToken.mutateAsync();
      setValidationResult(null);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="flex items-center justify-center py-8">
          <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center gap-3 mb-4">
        <Key className="w-5 h-5 text-gray-500" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
          HuggingFace Token
        </h2>
      </div>

      <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Your HuggingFace token is required to download models and push trained models to the Hub.
        Get your token from{' '}
        <a
          href="https://huggingface.co/settings/tokens"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-600 hover:underline inline-flex items-center gap-1"
        >
          huggingface.co/settings/tokens
          <ExternalLink className="w-3 h-3" />
        </a>
      </p>

      {tokenStatus?.has_token ? (
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-600 dark:text-gray-400">Current token:</span>
                <code className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-sm">
                  {tokenStatus.token_prefix}
                </code>
                {validationResult?.valid && (
                  <span className="flex items-center gap-1 text-green-600 text-sm">
                    <CheckCircle className="w-4 h-4" />
                    Valid ({validationResult.username})
                  </span>
                )}
              </div>
            </div>
            <button
              onClick={handleValidate}
              disabled={validateToken.isPending}
              className="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50"
            >
              {validateToken.isPending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                'Validate'
              )}
            </button>
            <button
              onClick={handleDelete}
              disabled={deleteToken.isPending}
              className="px-3 py-1.5 text-sm text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg disabled:opacity-50"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </div>

          {validationResult && !validationResult.valid && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-sm text-red-600 dark:text-red-400">
                {validationResult.error}
              </span>
            </div>
          )}

          {validationResult?.valid && (
            <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span className="text-sm font-medium text-green-600 dark:text-green-400">
                  Token is valid
                </span>
              </div>
              <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                <p>Username: {validationResult.username}</p>
                {validationResult.name && <p>Name: {validationResult.name}</p>}
                <p>Write access: {validationResult.can_write ? 'Yes' : 'No (read-only)'}</p>
              </div>
            </div>
          )}

          <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Replace with a new token:</p>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <input
                  type={showToken ? 'text' : 'password'}
                  value={token}
                  onChange={(e) => setToken(e.target.value)}
                  placeholder="hf_xxxxxxxxxxxxxxxx"
                  className="w-full px-3 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
                />
                <button
                  type="button"
                  onClick={() => setShowToken(!showToken)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <button
                onClick={handleSaveToken}
                disabled={!token.trim() || saveToken.isPending}
                className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {saveToken.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save'}
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                type={showToken ? 'text' : 'password'}
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="hf_xxxxxxxxxxxxxxxx"
                className="w-full px-3 py-2 pr-10 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
              />
              <button
                type="button"
                onClick={() => setShowToken(!showToken)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            <button
              onClick={handleSaveToken}
              disabled={!token.trim() || saveToken.isPending}
              className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {saveToken.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Save Token'}
            </button>
          </div>

          {validationResult && !validationResult.valid && (
            <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-sm text-red-600 dark:text-red-400">
                {validationResult.error}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PasswordSection() {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPasswords, setShowPasswords] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const changePassword = useChangePassword();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setSuccess(false);

    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }

    if (newPassword.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    try {
      await changePassword.mutateAsync({ current_password: currentPassword, new_password: newPassword });
      setSuccess(true);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to change password');
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex items-center gap-3 mb-4">
        <Lock className="w-5 h-5 text-gray-500" />
        <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Change Password</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Current Password
          </label>
          <input
            type={showPasswords ? 'text' : 'password'}
            value={currentPassword}
            onChange={(e) => setCurrentPassword(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            New Password
          </label>
          <input
            type={showPasswords ? 'text' : 'password'}
            value={newPassword}
            onChange={(e) => setNewPassword(e.target.value)}
            required
            minLength={8}
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Confirm New Password
          </label>
          <input
            type={showPasswords ? 'text' : 'password'}
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500"
          />
        </div>

        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="showPasswords"
            checked={showPasswords}
            onChange={(e) => setShowPasswords(e.target.checked)}
            className="rounded border-gray-300"
          />
          <label htmlFor="showPasswords" className="text-sm text-gray-600 dark:text-gray-400">
            Show passwords
          </label>
        </div>

        {error && (
          <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <AlertCircle className="w-5 h-5 text-red-500" />
            <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
          </div>
        )}

        {success && (
          <div className="flex items-center gap-2 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
            <CheckCircle className="w-5 h-5 text-green-500" />
            <span className="text-sm text-green-600 dark:text-green-400">
              Password changed successfully
            </span>
          </div>
        )}

        <button
          type="submit"
          disabled={changePassword.isPending}
          className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg disabled:opacity-50"
        >
          {changePassword.isPending ? (
            <span className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Changing...
            </span>
          ) : (
            'Change Password'
          )}
        </button>
      </form>
    </div>
  );
}
