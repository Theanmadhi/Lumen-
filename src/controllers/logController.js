import mongoose from "mongoose";

// Define the Log schema
const logSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    action: {
      type: String,
      enum: [
        "SUBSCRIBE",
        "UPGRADE",
        "DOWNGRADE",
        "DOWNGRADE_SCHEDULED",
        "CANCEL",
        "RENEW",
        "PAYMENT",
        "PAYMENT_FAILED",
        "REFUND",
      ],
      required: true,
    },
    entityType: {
      type: String,
      enum: ["SUBSCRIPTION", "PLAN", "BILLING", "USER"],
      required: true,
    },
    entityId: {
      type: mongoose.Schema.Types.ObjectId,
      required: true,
    },
    details: {
      type: Object,
      default: {},
    },
    ipAddress: String,
    userAgent: String,
  },
  { timestamps: true }
);

// Create Log model
const Log = mongoose.model("Log", logSchema);

/**
 * Create a new log entry
 * @param {Object} logData - The log data
 * @param {mongoose.ClientSession} session - Mongoose session for transactions
 */
export const createLog = async (logData, session = null) => {
  try {
    const logEntry = {
      user: logData.user,
      action: logData.action,
      entityType: logData.entityType,
      entityId: logData.entityId,
      details: logData.details || {},
      ipAddress: logData.ipAddress,
      userAgent: logData.userAgent,
    };

    if (session) {
      return await Log.create([logEntry], { session });
    } else {
      return await Log.create(logEntry);
    }
  } catch (error) {
    console.error("Error creating log entry:", error);
    throw error;
  }
};

/**
 * Get logs for a specific entity
 */
export const getEntityLogs = async (req, res) => {
  try {
    const { entityType, entityId } = req.params;
    
    const logs = await Log.find({
      entityType: entityType.toUpperCase(),
      entityId,
    })
      .populate("user", "username email")
      .sort({ createdAt: -1 });
    
    res.status(200).json({
      success: true,
      data: logs,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch logs",
      error: error.message,
    });
  }
};

/**
 * Get logs for a specific user
 */
export const getUserLogs = async (req, res) => {
  try {
    const userId = req.params.userId || req.user._id;
    
    const logs = await Log.find({ user: userId })
      .sort({ createdAt: -1 });
    
    res.status(200).json({
      success: true,
      data: logs,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Failed to fetch user logs",
      error: error.message,
    });
  }
};

export default { createLog, getEntityLogs, getUserLogs };