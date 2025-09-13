import dotenv from "dotenv";
dotenv.config();
import express from "express"
import cors from "cors"
import userRoutes from "./src/routes/userRoute.js"
import connectDB from "./src/config/dbconfig.js";

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({extended: true}));

const MONGO_URL = process.env.MONGO_URL;

app.use("/users", userRoutes);




connectDB()
    .then( ()=> {
        app.listen(process.env.PORT,
            ()=> console.log(`Server is successfully listening on port ${process.env.PORT}`)
        );
    })
    .catch((err) => {
        console.error('Failed to connect to the database:', err);
        process.exit(1);
    })